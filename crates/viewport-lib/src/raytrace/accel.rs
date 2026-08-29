//! Hardware acceleration structure for the ray-query traversal backend.
//!
//! Builds a bottom-level acceleration structure (BLAS) from the tracer's
//! world-space triangles and a single-instance top-level structure (TLAS) that
//! references it with the identity transform. The triangles are already in world
//! space and already reordered to match the `tris` storage buffer, so a triangle
//! at index `i` in the BLAS is `tris[i]` in the shader: `rayQuery`'s
//! `primitive_index` indexes straight into that buffer for normal/material
//! lookup, exactly as the compute traversal's `best_tri` did.
//!
//! Only compiled with the `raytrace-hardware` feature and only built on a device
//! that advertises [`RAY_QUERY_FEATURE`](crate::gpu::RAY_QUERY_FEATURE). It is
//! unsupported on Metal and the web, so it cannot be exercised on the primary dev
//! platform; the kernel that consumes it lives in `raytrace.wgsl`.

use crate::gpu::util::DeviceExt;

/// A built BLAS + TLAS pair, plus the vertex buffer the BLAS was built from.
///
/// The vertex buffer is kept alive for the lifetime of the structures because
/// the BLAS references it during the build; dropping it early is a use-after-free
/// on some backends.
pub(crate) struct HwAccel {
    /// Single top-level structure bound into the kernel at group 0, binding 13.
    pub(crate) tlas: crate::gpu::Tlas,
    // The BLAS is referenced by the TLAS instance, which keeps it alive, but hold
    // it explicitly so the ownership is obvious and the build inputs outlive use.
    _blas: crate::gpu::Blas,
    _vertex_buf: crate::gpu::Buffer,
}

impl HwAccel {
    /// Build the acceleration structures for `positions`: three vertices per
    /// triangle, tightly packed as `[f32; 3]` in the same order as the `tris`
    /// buffer. `positions.len()` must be a multiple of three and non-empty.
    pub(crate) fn build(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        positions: &[[f32; 3]],
    ) -> Self {
        debug_assert!(!positions.is_empty() && positions.len() % 3 == 0);
        let vertex_count = positions.len() as u32;

        // BLAS input geometry: opaque so `rayQuery` reports hits (a non-opaque
        // BLAS is un-hittable in WGSL until any-hit candidates exist), fast trace
        // since the tracer builds once and traces many times.
        let vertex_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_hw_vertices"),
            contents: bytemuck::cast_slice(positions),
            usage: crate::gpu::BufferUsages::BLAS_INPUT,
        });

        let geo_size = crate::gpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: crate::gpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: None,
            index_count: None,
            flags: crate::gpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &crate::gpu::CreateBlasDescriptor {
                label: Some("rt_hw_blas"),
                flags: crate::gpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: crate::gpu::AccelerationStructureUpdateMode::Build,
            },
            crate::gpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![geo_size.clone()],
            },
        );

        let mut tlas = device.create_tlas(&crate::gpu::CreateTlasDescriptor {
            label: Some("rt_hw_tlas"),
            max_instances: 1,
            flags: crate::gpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: crate::gpu::AccelerationStructureUpdateMode::Build,
        });
        // Identity 3x4 row-major affine: the triangles are already in world space.
        const IDENTITY_3X4: [f32; 12] = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        // mask 0xff so the kernel's 0xff cull mask always reports the instance.
        tlas[0] = Some(crate::gpu::TlasInstance::new(&blas, IDENTITY_3X4, 0, 0xff));

        // A BLAS may be built and referenced by a TLAS in the same submission.
        let geometry = crate::gpu::BlasTriangleGeometry {
            size: &geo_size,
            vertex_buffer: &vertex_buf,
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
            index_buffer: None,
            first_index: None,
            transform_buffer: None,
            transform_buffer_offset: None,
        };
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("rt_hw_accel_build"),
        });
        encoder.build_acceleration_structures(
            std::iter::once(&crate::gpu::BlasBuildEntry {
                blas: &blas,
                geometry: crate::gpu::BlasGeometries::TriangleGeometries(vec![geometry]),
            }),
            std::iter::once(&tlas),
        );
        queue.submit(std::iter::once(encoder.finish()));

        HwAccel {
            tlas,
            _blas: blas,
            _vertex_buf: vertex_buf,
        }
    }
}
