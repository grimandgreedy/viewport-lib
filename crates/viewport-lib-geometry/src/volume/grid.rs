//! A scalar field sampled on a regular 3D grid.

/// A structured 3D scalar field on a regular grid.
#[derive(Debug, Clone)]
pub struct VolumeData {
    /// Flattened scalar values in x-fastest order: `index = x + y*nx + z*nx*ny`.
    pub data: Vec<f32>,
    /// Grid dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// World-space origin of the grid corner `(0, 0, 0)`.
    pub origin: [f32; 3],
    /// Cell size in each axis direction.
    pub spacing: [f32; 3],
}

impl VolumeData {
    /// Check whether grid indices are within bounds.
    pub fn in_bounds(&self, ix: u32, iy: u32, iz: u32) -> bool {
        ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2]
    }

    /// Read the scalar value at grid point `(ix, iy, iz)`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn sample(&self, ix: u32, iy: u32, iz: u32) -> f32 {
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        self.data[ix as usize + iy as usize * nx + iz as usize * nx * ny]
    }
}

/// Trilinear interpolation of the volume at an arbitrary world-space position.
///
/// Returns the interpolated scalar value. Points outside the grid are clamped
/// to the boundary.
pub fn trilinear_sample(volume: &VolumeData, world_pos: [f32; 3]) -> f32 {
    let [nx, ny, nz] = volume.dims;

    // Convert world position to continuous grid coordinates.
    let gx = (world_pos[0] - volume.origin[0]) / volume.spacing[0];
    let gy = (world_pos[1] - volume.origin[1]) / volume.spacing[1];
    let gz = (world_pos[2] - volume.origin[2]) / volume.spacing[2];

    // Clamp to valid range.
    let gx = gx.clamp(0.0, (nx as f32) - 1.001);
    let gy = gy.clamp(0.0, (ny as f32) - 1.001);
    let gz = gz.clamp(0.0, (nz as f32) - 1.001);

    let ix = gx.floor() as u32;
    let iy = gy.floor() as u32;
    let iz = gz.floor() as u32;

    let fx = gx - ix as f32;
    let fy = gy - iy as f32;
    let fz = gz - iz as f32;

    let ix1 = (ix + 1).min(nx - 1);
    let iy1 = (iy + 1).min(ny - 1);
    let iz1 = (iz + 1).min(nz - 1);

    // 8-corner trilinear interpolation.
    let c000 = volume.sample(ix, iy, iz);
    let c100 = volume.sample(ix1, iy, iz);
    let c010 = volume.sample(ix, iy1, iz);
    let c110 = volume.sample(ix1, iy1, iz);
    let c001 = volume.sample(ix, iy, iz1);
    let c101 = volume.sample(ix1, iy, iz1);
    let c011 = volume.sample(ix, iy1, iz1);
    let c111 = volume.sample(ix1, iy1, iz1);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_data_sample() {
        let data = vec![
            1.0, 2.0, // z=0 row: (0,0,0)=1, (1,0,0)=2
            3.0, 4.0, // z=0 row: (0,1,0)=3, (1,1,0)=4
            5.0, 6.0, // z=1 row: (0,0,1)=5, (1,0,1)=6
            7.0, 8.0, // z=1 row: (0,1,1)=7, (1,1,1)=8
        ];
        let vol = VolumeData {
            data,
            dims: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        assert_eq!(vol.sample(0, 0, 0), 1.0);
        assert_eq!(vol.sample(1, 0, 0), 2.0);
        assert_eq!(vol.sample(0, 1, 0), 3.0);
        assert_eq!(vol.sample(1, 1, 0), 4.0);
        assert_eq!(vol.sample(0, 0, 1), 5.0);
        assert_eq!(vol.sample(1, 0, 1), 6.0);
        assert_eq!(vol.sample(0, 1, 1), 7.0);
        assert_eq!(vol.sample(1, 1, 1), 8.0);
    }

    #[test]
    fn test_trilinear_interpolation() {
        let data = vec![
            0.0, 1.0, // (0,0,0)=0, (1,0,0)=1
            0.0, 1.0, // (0,1,0)=0, (1,1,0)=1
            0.0, 1.0, // (0,0,1)=0, (1,0,1)=1
            0.0, 1.0, // (0,1,1)=0, (1,1,1)=1
        ];
        let vol = VolumeData {
            data,
            dims: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        // At grid points.
        let v00 = trilinear_sample(&vol, [0.0, 0.0, 0.0]);
        assert!(
            (v00 - 0.0).abs() < 0.01,
            "Expected 0.0 at origin, got {}",
            v00
        );

        let v10 = trilinear_sample(&vol, [0.999, 0.0, 0.0]);
        assert!(
            (v10 - 1.0).abs() < 0.02,
            "Expected ~1.0 at (1,0,0), got {}",
            v10
        );

        // At midpoint along X: linear gradient means value = 0.5.
        let mid = trilinear_sample(&vol, [0.5, 0.0, 0.0]);
        assert!(
            (mid - 0.5).abs() < 0.01,
            "Expected 0.5 at midpoint, got {}",
            mid
        );

        // At midpoint in all axes.
        let center = trilinear_sample(&vol, [0.5, 0.5, 0.5]);
        assert!(
            (center - 0.5).abs() < 0.01,
            "Expected 0.5 at center, got {}",
            center
        );
    }
}
