//! Procedural geometry helper functions shared across showcase scenes.

use viewport_lib::MeshData;

/// Generate a UV sphere with positions, normals, UVs, and triangulated indices.
pub fn make_uv_sphere(lon_segs: usize, lat_segs: usize, radius: f32) -> MeshData {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut tangents: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for lat in 0..=lat_segs {
        let theta = std::f32::consts::PI * lat as f32 / lat_segs as f32;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for lon in 0..=lon_segs {
            let phi = 2.0 * std::f32::consts::PI * lon as f32 / lon_segs as f32;
            let sin_p = phi.sin();
            let cos_p = phi.cos();
            let x = sin_t * cos_p;
            let y = cos_t;
            let z = sin_t * sin_p;
            positions.push([x * radius, y * radius, z * radius]);
            normals.push([x, y, z]);
            uvs.push([lon as f32 / lon_segs as f32, lat as f32 / lat_segs as f32]);
            tangents.push([-sin_p, 0.0, cos_p, 1.0]);
        }
    }

    let stride = lon_segs + 1;
    for lat in 0..lat_segs {
        for lon in 0..lon_segs {
            let a = (lat * stride + lon) as u32;
            let b = ((lat + 1) * stride + lon) as u32;
            let c = ((lat + 1) * stride + lon + 1) as u32;
            let d = (lat * stride + lon + 1) as u32;
            if lat > 0 {
                indices.extend_from_slice(&[a, d, b]);
            }
            if lat < lat_segs - 1 {
                indices.extend_from_slice(&[b, d, c]);
            }
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh.uvs = Some(uvs);
    mesh.tangents = Some(tangents);
    mesh
}

/// Generate a box mesh with positions, normals, UVs (0..1 per face), and indices.
pub fn make_box_with_uvs(w: f32, h: f32, d: f32) -> MeshData {
    let hx = w * 0.5;
    let hy = h * 0.5;
    let hz = d * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let face_uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut push_face = |verts: [[f32; 3]; 4], normal: [f32; 3]| {
        let base = positions.len() as u32;
        for (v, uv) in verts.iter().zip(face_uvs.iter()) {
            positions.push(*v);
            normals.push(normal);
            uvs.push(*uv);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    push_face(
        [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]],
        [1.0, 0.0, 0.0],
    );
    push_face(
        [
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-hx, -hy, -hz],
        ],
        [-1.0, 0.0, 0.0],
    );
    push_face(
        [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]],
        [0.0, 1.0, 0.0],
    );
    push_face(
        [
            [-hx, -hy, hz],
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
        ],
        [0.0, -1.0, 0.0],
    );
    push_face(
        [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        [0.0, 0.0, 1.0],
    );
    push_face(
        [
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
        ],
        [0.0, 0.0, -1.0],
    );

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh.uvs = Some(uvs);
    mesh
}

/// Procedural brick normal map (RGBA Unorm8).
pub fn make_brick_normal_map(w: u32, h: u32) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for row in 0..h {
        for col in 0..w {
            let u = col as f32 / w as f32;
            let v = row as f32 / h as f32;
            let brick_row = (v * 4.0) as u32;
            let offset = if brick_row % 2 == 0 { 0.0 } else { 0.5 };
            let bx = ((u + offset) * 6.0).fract();
            let by = (v * 4.0).fract();
            let mortar_x = bx < 0.1 || bx > 0.9;
            let mortar_y = by < 0.1 || by > 0.9;
            let (nx, ny, nz) = if mortar_x || mortar_y {
                (0.0f32, 0.0f32, 1.0f32)
            } else {
                let cx = (bx - 0.5) * 2.0;
                let cy = (by - 0.5) * 2.0;
                let tx = -cx * 0.6;
                let ty = -cy * 0.6;
                let len = (tx * tx + ty * ty + 1.0f32).sqrt();
                (tx / len, ty / len, 1.0 / len)
            };
            let i = ((row * w + col) * 4) as usize;
            data[i] = ((nx * 0.5 + 0.5) * 255.0) as u8;
            data[i + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            data[i + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
    data
}

/// Procedural brick AO map (RGBA8 Unorm).
pub fn make_brick_ao_map(w: u32, h: u32) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for row in 0..h {
        for col in 0..w {
            let u = col as f32 / w as f32;
            let v = row as f32 / h as f32;
            let brick_row = (v * 4.0) as u32;
            let offset = if brick_row % 2 == 0 { 0.0 } else { 0.5 };
            let bx = ((u + offset) * 6.0).fract();
            let by = (v * 4.0).fract();
            let dx = 0.5 - (bx - 0.5).abs();
            let dy = 0.5 - (by - 0.5).abs();
            let edge_dist = dx.min(dy);
            let t = (1.0 - edge_dist / 0.12).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            let ao = ((1.0 - 0.65 * t) * 255.0) as u8;
            let i = ((row * w + col) * 4) as usize;
            data[i] = ao;
            data[i + 1] = ao;
            data[i + 2] = ao;
            data[i + 3] = 255;
        }
    }
    data
}

/// Procedural tile normal map (RGBA Unorm8).
pub fn make_tile_normal_map(w: u32, h: u32) -> Vec<u8> {
    let tiles = 4.0f32;
    let bevel = 0.1f32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for row in 0..h {
        for col in 0..w {
            let u = col as f32 / w as f32;
            let v = row as f32 / h as f32;
            let tx = (u * tiles).fract();
            let ty = (v * tiles).fract();
            let left = tx;
            let right = 1.0 - tx;
            let bot = ty;
            let top = 1.0 - ty;
            let nx = if left < bevel {
                -(1.0 - left / bevel) * 0.7
            } else if right < bevel {
                (1.0 - right / bevel) * 0.7
            } else {
                0.0f32
            };
            let ny = if bot < bevel {
                -(1.0 - bot / bevel) * 0.7
            } else if top < bevel {
                (1.0 - top / bevel) * 0.7
            } else {
                0.0f32
            };
            let len = (nx * nx + ny * ny + 1.0f32).sqrt();
            let (nx, ny, nz) = (nx / len, ny / len, 1.0 / len);
            let i = ((row * w + col) * 4) as usize;
            data[i] = ((nx * 0.5 + 0.5) * 255.0) as u8;
            data[i + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            data[i + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
    data
}

/// Procedural tile AO map (RGBA8 Unorm).
pub fn make_tile_ao_map(w: u32, h: u32) -> Vec<u8> {
    let tiles = 4.0f32;
    let groove = 0.07f32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for row in 0..h {
        for col in 0..w {
            let u = col as f32 / w as f32;
            let v = row as f32 / h as f32;
            let tx = (u * tiles).fract();
            let ty = (v * tiles).fract();
            let dx = tx.min(1.0 - tx);
            let dy = ty.min(1.0 - ty);
            let edge_dist = dx.min(dy);
            let t = (1.0 - edge_dist / groove).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            let ao = ((1.0 - 0.75 * t) * 255.0) as u8;
            let i = ((row * w + col) * 4) as usize;
            data[i] = ao;
            data[i + 1] = ao;
            data[i + 2] = ao;
            data[i + 3] = 255;
        }
    }
    data
}
