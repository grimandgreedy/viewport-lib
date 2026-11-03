//! Print this crate's composed shaders as JSON, for the browser shader check.
//!
//! The bodies under `src/` are fragments: the group-0 declarations and the
//! shading helpers are spliced in at pipeline build time, so the only way to
//! hand a validator the real source is to ask the crate to compose it.

fn main() {
    let sources = viewport_lib_plugins::shader_sources();
    let mut out = String::from("{");
    for (i, (name, source)) in sources.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{}:{}", json_string(name), json_string(source)));
    }
    out.push('}');
    println!("{out}");
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
