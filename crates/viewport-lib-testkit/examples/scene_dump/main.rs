// Render catalogue scenes sequentially in ONE harness and dump each render,
// to reproduce cross-scene state leaks the snapshot loop exposed.
use viewport_lib_testkit::{Harness, frame_for, scenes};

fn main() {
    let out_dir = std::env::args().nth(1).expect("out dir");
    let names: Vec<String> = std::env::args().skip(2).collect();
    let mut h = Harness::new().expect("harness");
    for name in &names {
        let scene = scenes::scene_by_name(name).expect("scene");
        let built = h.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [400.0, 300.0]);
        let _ = h.render(&frame, 400, 300);
        let px = h.render(&frame, 400, 300);
        let img = viewport_lib_testkit::golden::RgbaImage::from_raw(400, 300, px).unwrap();
        let path = format!("{out_dir}/{name}.seq.png");
        img.save(&path).unwrap();
        println!("wrote {path}");
    }
}
