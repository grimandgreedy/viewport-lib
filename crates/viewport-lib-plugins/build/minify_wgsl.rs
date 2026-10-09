// WGSL comment and indentation stripping for the `minify-shaders` feature.
//
// This file is shared by the build scripts of viewport-lib,
// viewport-lib-plugins and viewport-lib-post-effects. Each crate keeps an
// identical copy under `build/` (a published crate cannot reach a sibling's
// files), and a viewport-lib test checks the copies have not drifted.

/// Strip line comments, blank lines and leading and trailing whitespace from
/// WGSL source.
///
/// Comment lines the renderer reads at runtime are kept whole: a full-line
/// comment whose text starts with `<` (region tags such as
/// `// <viewport-shade-slot:...>`), `#` (`// #include`), `@`
/// (`// @viewport-wgsl-version`), `BEGIN_` or `END_`.
/// A kept line also keeps any code after its tag, since some tags turn the
/// rest of their line back into code when removed.
///
/// Whitespace inside a line is never touched, so runtime rewrites that match
/// a statement or an aligned declaration block still find it. WGSL has no
/// string literals, so `//` outside a block comment always starts a comment;
/// block comments are left as they are.
pub fn minify_wgsl(source: &str) -> String {
    let mut out = String::with_capacity(source.len() / 2);
    let mut in_block = 0usize;
    for line in source.lines() {
        let t = line.trim();
        if in_block == 0
            && let Some(body) = t.strip_prefix("//")
        {
            if is_marker(body) {
                out.push_str(t);
                out.push('\n');
            }
            continue;
        }
        if in_block > 0 || t.contains("/*") {
            // Rare enough to pass through untouched rather than parse.
            in_block = block_depth_after(t, in_block);
            if !t.is_empty() {
                out.push_str(t);
                out.push('\n');
            }
            continue;
        }
        let code = match t.find("//") {
            Some(i) => t[..i].trim_end(),
            None => t,
        };
        if !code.is_empty() {
            out.push_str(code);
            out.push('\n');
        }
    }
    out
}

fn is_marker(comment_body: &str) -> bool {
    let b = comment_body.trim_start();
    b.starts_with('<')
        || b.starts_with('#')
        || b.starts_with('@')
        || b.starts_with("BEGIN_")
        || b.starts_with("END_")
}

// Track nesting of `/* */` comments across a line, ignoring `//` inside them.
fn block_depth_after(line: &str, mut depth: usize) -> usize {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        match (bytes[i], bytes[i + 1]) {
            (b'/', b'*') => {
                depth += 1;
                i += 2;
            }
            (b'*', b'/') if depth > 0 => {
                depth -= 1;
                i += 2;
            }
            (b'/', b'/') if depth == 0 => break,
            _ => i += 1,
        }
    }
    depth
}
