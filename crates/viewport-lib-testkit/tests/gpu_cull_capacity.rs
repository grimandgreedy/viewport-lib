//! The cull pass packs its visible list in instance order, so the same scene
//! draws in the same order every frame. A submission with more batches than
//! the fixed chunk plan holds falls back to packing in the order threads
//! finish, which is still a correct list, just not a reproducible one.
//!
//! Both branches are exercised here: one submission just under the batch
//! ceiling, which must come back in exact instance order, and one just over
//! it, which must come back as a permutation with the right contents. The
//! over-capacity branch has no other coverage, and nothing else would notice
//! if it stopped producing a usable list.

use viewport_lib::plugin_api::CullSubmission;
use viewport_lib::wgpu;
use viewport_lib_testkit::Harness;

/// Matches `MAX_PLAN_BATCHES` in the renderer's cull resources.
const MAX_PLAN_BATCHES: u32 = 8192;

/// Runs one cull submission and reads back the visible list and the per-batch
/// instance counts from the indirect entries.
///
/// `sizes` gives each batch's instance count, so a caller can mix batch sizes
/// and include empty batches. An instance is placed inside the frustum when
/// `keep` returns true for its index and far outside it otherwise, so the
/// caller controls exactly who survives.
fn run_cull(h: &mut Harness, sizes: &[u32], keep: impl Fn(u32) -> bool) -> (Vec<u32>, Vec<u32>) {
    let n = sizes.len();
    let batch_count = n as u32;
    let instance_count: u32 = sizes.iter().sum();

    // `InstanceAabb` and `BatchMeta` are both 32 bytes; written out by hand so
    // the test does not need a bytemuck dependency of its own.
    let mut aabb_bytes: Vec<u8> = Vec::with_capacity(instance_count as usize * 32);
    let mut meta_bytes: Vec<u8> = Vec::with_capacity(n * 32);
    let mut first = 0u32;
    for (b, &size) in sizes.iter().enumerate() {
        for k in 0..size {
            let i = first + k;
            // Culled instances sit well outside the orthographic box below.
            let centre = if keep(i) { 0.0f32 } else { 1.0e6 };
            for _ in 0..3 {
                aabb_bytes.extend_from_slice(&(centre - 0.5).to_le_bytes());
            }
            aabb_bytes.extend_from_slice(&(b as u32).to_le_bytes()); // batch_index
            for _ in 0..3 {
                aabb_bytes.extend_from_slice(&(centre + 0.5).to_le_bytes());
            }
            aabb_bytes.extend_from_slice(&1u32.to_le_bytes()); // cast_shadows
        }
        // index_count, first_index, instance_offset, instance_count,
        // vis_offset, is_transparent, base_vertex, _reserved_flags.
        for w in [36u32, 0, first, size, first, 0, 0, 0] {
            meta_bytes.extend_from_slice(&w.to_le_bytes());
        }
        first += size;
    }

    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let aabb_buf = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_aabbs"),
        size: aabb_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let meta_buf = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_metas"),
        size: meta_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    h.queue.write_buffer(&aabb_buf, 0, &aabb_bytes);
    h.queue.write_buffer(&meta_buf, 0, &meta_bytes);
    let counter = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_counter"),
        size: u64::from(batch_count) * 4,
        usage: storage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let visible = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_visible"),
        size: u64::from(instance_count) * 4,
        usage: storage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indirect = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_indirect"),
        size: u64::from(batch_count) * 20,
        usage: storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Pre-fill the visible list with a sentinel so a slot the cull never wrote
    // is distinguishable from a legitimate instance index of 0.
    h.queue
        .write_buffer(&visible, 0, &vec![0xffu8; instance_count as usize * 4]);
    h.queue.write_buffer(&counter, 0, &vec![0u8; n * 4]);

    let stage_visible = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_visible_stage"),
        size: u64::from(instance_count) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_indirect = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capacity_indirect_stage"),
        size: u64::from(batch_count) * 20,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // A frustum wide enough that nothing is culled: the point here is the
    // packing, not the reject.
    let view_proj = glam::Mat4::orthographic_rh(-1e4, 1e4, -1e4, 1e4, -1e4, 1e4);
    let frustum = viewport_lib::camera::frustum::Frustum::from_view_proj(&view_proj);

    let sub = CullSubmission {
        instance_aabbs: &aabb_buf,
        instance_count,
        batch_meta: &meta_buf,
        batch_count,
        counter: &counter,
        visible_out: &visible,
        indirect_out: &indirect,
        shadow_pass: false,
    };

    let mut encoder = h
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capacity_cull"),
        });
    h.renderer
        .submit_cull(&h.device, &h.queue, &mut encoder, &frustum, &sub);
    encoder.copy_buffer_to_buffer(
        &visible,
        0,
        &stage_visible,
        0,
        u64::from(instance_count) * 4,
    );
    encoder.copy_buffer_to_buffer(
        &indirect,
        0,
        &stage_indirect,
        0,
        u64::from(batch_count) * 20,
    );
    h.queue.submit(Some(encoder.finish()));

    let read = |buf: &wgpu::Buffer| -> Vec<u8> {
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        h.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("poll");
        let bytes = slice.get_mapped_range().to_vec();
        buf.unmap();
        bytes
    };

    let vis_bytes = read(&stage_visible);
    let vis: Vec<u32> = (0..instance_count as usize)
        .map(|i| u32::from_le_bytes(vis_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();
    // Read only the instance_count field of each 20-byte indirect entry.
    let ind_bytes = read(&stage_indirect);
    let counts: Vec<u32> = (0..n)
        .map(|i| u32::from_le_bytes(ind_bytes[i * 20 + 4..i * 20 + 8].try_into().unwrap()))
        .collect();
    (vis, counts)
}

#[test]
fn fixture_cull_packs_survivors_in_instance_order() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    // One batch of 2048 instances spans several compaction chunks, so this
    // covers a chunk deriving its base from its predecessors' counts rather
    // than only the trivial first-chunk case. Every third instance survives,
    // which leaves a different count in each chunk.
    let per_batch = 2048;
    let keep = |i: u32| i % 3 == 0;
    let (vis, counts) = run_cull(&mut h, &[per_batch], keep);

    let expected: Vec<u32> = (0..per_batch).filter(|&i| keep(i)).collect();
    assert_eq!(
        counts[0],
        expected.len() as u32,
        "indirect entry disagrees with the number of survivors"
    );
    assert_eq!(
        &vis[..expected.len()],
        &expected[..],
        "survivors were not packed in instance order"
    );
}

#[test]
fn fixture_cull_within_plan_capacity_draws_every_batch() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    let batches = MAX_PLAN_BATCHES;
    let (vis, counts) = run_cull(&mut h, &vec![1; batches as usize], |_| true);

    let expected: Vec<u32> = (0..batches).collect();
    assert_eq!(
        vis, expected,
        "compaction did not pack the visible list in instance order"
    );
    assert!(
        counts.iter().all(|&c| c == 1),
        "expected one visible instance per batch, got {:?}",
        &counts[..counts.len().min(8)]
    );
}

#[test]
fn fixture_cull_over_plan_capacity_still_draws_every_instance() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    // One batch past the plan, so the compaction is skipped and the cull
    // kernel's own arrival-order list is what gets drawn.
    let batches = MAX_PLAN_BATCHES + 1;
    let (mut vis, counts) = run_cull(&mut h, &vec![1; batches as usize], |_| true);

    assert!(
        counts.iter().all(|&c| c == 1),
        "expected one visible instance per batch on the fallback path"
    );
    // Each batch owns one visibility slot here, so the fallback list happens to
    // match instance order too. What matters is that every slot was written and
    // the contents are the full set.
    vis.sort_unstable();
    let expected: Vec<u32> = (0..batches).collect();
    assert_eq!(
        vis, expected,
        "fallback path left the visible list incomplete"
    );
}

#[test]
fn fixture_cull_resolves_chunk_owner_across_mixed_and_empty_batches() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    // A chunk finds its batch by bisecting the chunk plan. Empty batches share
    // a plan entry with whatever follows them, and batches whose instance count
    // is not a multiple of the chunk size leave a partly filled last chunk, so
    // both go in here along with runs of consecutive empties and an empty tail.
    let sizes: Vec<u32> = vec![300, 0, 0, 512, 1, 0, 700, 256, 0, 0];
    let total: u32 = sizes.iter().sum();
    let keep = |i: u32| i % 5 != 0;
    let (vis, counts) = run_cull(&mut h, &sizes, keep);

    let mut first = 0u32;
    for (b, &size) in sizes.iter().enumerate() {
        let expected: Vec<u32> = (first..first + size).filter(|&i| keep(i)).collect();
        assert_eq!(
            counts[b],
            expected.len() as u32,
            "batch {b} (size {size}) reported the wrong visible count"
        );
        let got = &vis[first as usize..first as usize + expected.len()];
        assert_eq!(
            got,
            &expected[..],
            "batch {b} (size {size}) was not packed in instance order"
        );
        first += size;
    }
    assert_eq!(first, total);
}

#[test]
fn fixture_cull_plans_chunks_across_workgroup_tiles() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    // The chunk plan is a prefix sum over batches, run as a workgroup scan over
    // tiles of 256 batches with a carry between them. This spans three tiles
    // and does not end on a tile boundary, and the batches alternate between
    // one chunk and two so the carry is different at every tile: a scan that
    // dropped or mis-ordered the carry would put later batches' survivors at
    // the wrong offsets rather than losing them, which the per-batch assertions
    // below catch.
    let sizes: Vec<u32> = (0..520).map(|b| if b % 2 == 0 { 1 } else { 257 }).collect();
    let keep = |i: u32| i % 4 != 0;
    let (vis, counts) = run_cull(&mut h, &sizes, keep);

    let mut first = 0u32;
    for (b, &size) in sizes.iter().enumerate() {
        let expected: Vec<u32> = (first..first + size).filter(|&i| keep(i)).collect();
        assert_eq!(
            counts[b],
            expected.len() as u32,
            "batch {b} (size {size}) reported the wrong visible count"
        );
        assert_eq!(
            &vis[first as usize..first as usize + expected.len()],
            &expected[..],
            "batch {b} (size {size}) landed at the wrong offset"
        );
        first += size;
    }
}
