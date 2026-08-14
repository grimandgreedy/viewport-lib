//! World-space triangle BVH for the path tracer.
//!
//! Binned-SAH build on the CPU, flattened to a depth-first node array whose
//! layout matches `Node` in `raytrace.wgsl`: the left child of an interior node
//! is the next node in the array, and `left_first` holds the right child index;
//! for a leaf, `count > 0` and `left_first` is the first triangle slot in the
//! reordered triangle array.

use glam::Vec3;

/// GPU BVH node, 32 bytes, matching the WGSL `Node` struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuNode {
    aabb_min: [f32; 3],
    left_first: u32,
    aabb_max: [f32; 3],
    count: u32,
}

const LEAF_SIZE: usize = 4;
const NUM_BINS: usize = 12;

struct Prim {
    centroid: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    index: u32,
}

/// Build a BVH over the given per-triangle positions (mesh-local for a BLAS, or
/// world space for the legacy single-level tree).
///
/// Returns the flat node array and the triangle ordering: `order[k]` is the
/// original triangle index that occupies slot `k` in the reordered triangle
/// array the leaves reference.
pub(crate) fn build(tris: &[[Vec3; 3]]) -> (Vec<GpuNode>, Vec<u32>) {
    let prims: Vec<Prim> = tris
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let bmin = t[0].min(t[1]).min(t[2]);
            let bmax = t[0].max(t[1]).max(t[2]);
            Prim {
                centroid: (bmin + bmax) * 0.5,
                bmin,
                bmax,
                index: i as u32,
            }
        })
        .collect();
    build_prims(prims, tris.len())
}

/// Build a BVH over a set of world-space AABBs. Used for the top-level structure
/// (TLAS): one primitive per instance, its bounds the instance's world AABB.
/// `order[k]` is the original instance index occupying leaf slot `k`.
pub(crate) fn build_bounds(bounds: &[(Vec3, Vec3)]) -> (Vec<GpuNode>, Vec<u32>) {
    let prims: Vec<Prim> = bounds
        .iter()
        .enumerate()
        .map(|(i, &(bmin, bmax))| Prim {
            centroid: (bmin + bmax) * 0.5,
            bmin,
            bmax,
            index: i as u32,
        })
        .collect();
    build_prims(prims, bounds.len())
}

/// Flatten a prepared primitive list into a node array + reordering. Shared by
/// the triangle (BLAS) and bounds (TLAS) builders.
fn build_prims(mut prims: Vec<Prim>, hint: usize) -> (Vec<GpuNode>, Vec<u32>) {
    let mut nodes: Vec<GpuNode> = Vec::with_capacity(hint.max(1) * 2);
    if prims.is_empty() {
        nodes.push(GpuNode {
            aabb_min: [0.0; 3],
            left_first: 0,
            aabb_max: [0.0; 3],
            count: 0,
        });
        return (nodes, Vec::new());
    }

    let n = prims.len();
    build_range(&mut prims, 0, n, &mut nodes);

    let order = prims.iter().map(|p| p.index).collect();
    (nodes, order)
}

/// Shift a freshly built BLAS so it can be concatenated behind `node_base`
/// other nodes with its triangles behind `tri_base` other triangles. An
/// interior node's right-child index moves by `node_base` (its implicit
/// self+1 left child shifts with it); a leaf's first-triangle slot moves by
/// `tri_base`. The TLAS sits at the front of the combined array and needs no
/// rebase.
pub(crate) fn rebase(nodes: &mut [GpuNode], node_base: u32, tri_base: u32) {
    for n in nodes.iter_mut() {
        if n.count > 0 {
            n.left_first += tri_base;
        } else {
            n.left_first += node_base;
        }
    }
}

fn node_bounds(prims: &[Prim], start: usize, end: usize) -> (Vec3, Vec3) {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for p in &prims[start..end] {
        bmin = bmin.min(p.bmin);
        bmax = bmax.max(p.bmax);
    }
    (bmin, bmax)
}

fn centroid_bounds(prims: &[Prim], start: usize, end: usize) -> (Vec3, Vec3) {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for p in &prims[start..end] {
        bmin = bmin.min(p.centroid);
        bmax = bmax.max(p.centroid);
    }
    (bmin, bmax)
}

fn area(bmin: Vec3, bmax: Vec3) -> f32 {
    let d = (bmax - bmin).max(Vec3::ZERO);
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

/// Build the subtree for `prims[start..end]`, returning the node's index.
fn build_range(prims: &mut [Prim], start: usize, end: usize, nodes: &mut Vec<GpuNode>) -> u32 {
    let node_index = nodes.len() as u32;
    let (bmin, bmax) = node_bounds(prims, start, end);
    nodes.push(GpuNode {
        aabb_min: bmin.to_array(),
        left_first: 0,
        aabb_max: bmax.to_array(),
        count: 0,
    });

    let count = end - start;
    let make_leaf = |nodes: &mut Vec<GpuNode>| {
        nodes[node_index as usize].left_first = start as u32;
        nodes[node_index as usize].count = count as u32;
    };

    if count <= LEAF_SIZE {
        make_leaf(nodes);
        return node_index;
    }

    let (cmin, cmax) = centroid_bounds(prims, start, end);
    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let axis_extent = extent[axis];
    if axis_extent < 1.0e-8 {
        make_leaf(nodes);
        return node_index;
    }

    // Binned SAH over the chosen axis.
    let mut bin_count = [0u32; NUM_BINS];
    let mut bin_min = [Vec3::splat(f32::INFINITY); NUM_BINS];
    let mut bin_max = [Vec3::splat(f32::NEG_INFINITY); NUM_BINS];
    let scale = NUM_BINS as f32 / axis_extent;
    let bin_of = |c: &Prim| -> usize {
        let b = ((c.centroid[axis] - cmin[axis]) * scale) as usize;
        b.min(NUM_BINS - 1)
    };
    for p in &prims[start..end] {
        let b = bin_of(p);
        bin_count[b] += 1;
        bin_min[b] = bin_min[b].min(p.bmin);
        bin_max[b] = bin_max[b].max(p.bmax);
    }

    // Sweep to get left/right cost for each of the NUM_BINS-1 split planes.
    let mut left_area = [0.0f32; NUM_BINS - 1];
    let mut left_count = [0u32; NUM_BINS - 1];
    {
        let mut cmn = Vec3::splat(f32::INFINITY);
        let mut cmx = Vec3::splat(f32::NEG_INFINITY);
        let mut cnt = 0u32;
        for i in 0..NUM_BINS - 1 {
            cnt += bin_count[i];
            cmn = cmn.min(bin_min[i]);
            cmx = cmx.max(bin_max[i]);
            left_count[i] = cnt;
            left_area[i] = if cnt > 0 { area(cmn, cmx) } else { 0.0 };
        }
    }
    let mut best_cost = f32::INFINITY;
    let mut best_split = 0usize;
    {
        let mut cmn = Vec3::splat(f32::INFINITY);
        let mut cmx = Vec3::splat(f32::NEG_INFINITY);
        let mut cnt = 0u32;
        for i in (0..NUM_BINS - 1).rev() {
            cnt += bin_count[i + 1];
            cmn = cmn.min(bin_min[i + 1]);
            cmx = cmx.max(bin_max[i + 1]);
            let right_area = if cnt > 0 { area(cmn, cmx) } else { 0.0 };
            let cost = left_area[i] * left_count[i] as f32 + right_area * cnt as f32;
            if cost > 0.0 && cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }
    }

    // Partition prims into [start, mid) and [mid, end) by bin.
    let split_bin = best_split;
    let mut mid;
    {
        let mut i = start;
        let mut j = end;
        // Simple stable-ish partition using swaps.
        while i < j {
            if bin_of(&prims[i]) <= split_bin {
                i += 1;
            } else {
                j -= 1;
                prims.swap(i, j);
            }
        }
        mid = i;
    }

    // Guard against a degenerate partition (everything on one side): fall back
    // to a median split so recursion always shrinks.
    if mid == start || mid == end {
        mid = start + count / 2;
        prims[start..end].sort_unstable_by(|a, b| {
            a.centroid[axis]
                .partial_cmp(&b.centroid[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Left child is the next node pushed; record the right child index.
    let _left = build_range(prims, start, mid, nodes);
    let right = build_range(prims, mid, end, nodes);
    nodes[node_index as usize].left_first = right;
    nodes[node_index as usize].count = 0;
    node_index
}
