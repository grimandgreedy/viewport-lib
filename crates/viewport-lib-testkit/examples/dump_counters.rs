//! Throwaway: print the deterministic FrameStats counters for every catalogue
//! scene (second frame, settled). Used to seed the literals in the
//! `scene_counters` test, then removed.

use viewport_lib_testkit::{Harness, catalogue, frame_for};

fn main() {
    let Some(mut h) = Harness::new() else {
        eprintln!("no GPU adapter");
        return;
    };
    let (w, hgt) = (320u32, 240u32);
    println!(
        "scene,total_objects,visible_objects,draw_calls,instanced_batches,per_object_items,per_object_bgs,triangles,batches_reup,batches_skip"
    );
    for scene in catalogue() {
        let built = h.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, hgt as f32]);
        let s = h.render_two_frames(&frame, w, hgt);
        println!(
            "{},{},{},{},{},{},{},{},{},{}",
            scene.name,
            s.total_objects,
            s.visible_objects,
            s.draw_calls,
            s.instanced_batches,
            s.per_object_items,
            s.per_object_bind_groups_built,
            s.triangles_submitted,
            s.batches_reuploaded,
            s.batches_skipped,
        );
    }

    // Geometry-slab and buffer-bind counters: how the shared slab collapses
    // per-mesh binds. These are the structural buffer invariants (a regression
    // that stops sharing the slab, or reinstates per-batch binds, shows here).
    // Each on a fresh harness so slab residency reflects that scene alone.
    println!("\n-- slab / buffer binds (settled, one scene per harness) --");
    println!("scene,slab_chunks,main_binds,main_draw_cmds,shadow_binds,shadow_draw_cmds");
    for scene in catalogue() {
        let Some(mut hh) = Harness::new() else {
            break;
        };
        let built = hh.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, hgt as f32]);
        let s = hh.render_two_frames(&frame, w, hgt);
        println!(
            "{},{},{},{},{},{}",
            scene.name,
            s.slab_chunk_count,
            s.main_buffer_binds,
            s.main_draw_commands,
            s.shadow_buffer_binds,
            s.shadow_draw_commands,
        );
    }

    // Camera-motion probe: settle at camera 0, then move to camera 1 and read
    // whether static geometry was needlessly re-uploaded.
    println!("\n-- camera motion (move should not re-upload static batches) --");
    println!("scene,reup_after_move,bgs_after_move");
    for scene in catalogue() {
        if scene.cameras.len() < 2 {
            continue;
        }
        let built = h.build_scene(&scene);
        let f0 = frame_for(&built, &scene.cameras[0].camera, [w as f32, hgt as f32]);
        let _ = h.render_two_frames(&f0, w, hgt);
        let f1 = frame_for(&built, &scene.cameras[1].camera, [w as f32, hgt as f32]);
        let s = {
            let _ = h.render(&f1, w, hgt);
            h.stats()
        };
        println!(
            "{},{},{}",
            scene.name, s.batches_reuploaded, s.per_object_bind_groups_built
        );
    }
}
