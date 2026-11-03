//! Every depth compare and depth clear in the library and the built-in plugins
//! goes through `plugin_api::depth`, so the scene depth convention is set in
//! one place. A raw compare direction or clear value outside that module is a
//! site that would not follow a change of convention.

use std::path::{Path, PathBuf};

const FORBIDDEN: &[&str] = &[
    "CompareFunction::Less",
    "CompareFunction::Greater",
    "LoadOp::Clear(1.0)",
    "LoadOp::Clear(0.0)",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read source dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn depth_compares_and_clears_use_the_convention_module() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        manifest.join("src"),
        manifest.join("../viewport-lib-plugins/src"),
        manifest.join("../viewport-lib-post-effects/src"),
    ];
    let convention = manifest.join("src/plugin_api/depth.rs");
    let mut files = Vec::new();
    for root in &roots {
        if root.is_dir() {
            rust_files(root, &mut files);
        }
    }
    assert!(files.len() > 100, "found only {} source files", files.len());

    let mut hits = Vec::new();
    for file in &files {
        if file == &convention {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("read source file");
        for (n, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            if FORBIDDEN.iter().any(|f| code.contains(f)) {
                hits.push(format!("{}:{}: {}", file.display(), n + 1, code));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "raw depth compare or clear outside plugin_api::depth:\n{}",
        hits.join("\n")
    );
}
