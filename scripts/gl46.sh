#!/usr/bin/env bash
# Run cargo in the testkit with wgpu-hal patched to ask EGL for a desktop
# GL 4.6 core context instead of 3.3, and WGPU_BACKEND=gl set.
#
# Stock wgpu-hal asks for exactly 3.3 on the desktop GL path; NVIDIA's driver
# returns exactly 3.3, which has no compute, so request_device fails and every
# testkit GPU test skips. 4.6 is NVIDIA's highest core version and clears
# wgpu-hal's 4.3 storage and compute checks.
#
# Usage, from anywhere (TESTKIT defaults to this checkout's testkit):
#   gl46.sh <leg> <cargo args...>
#   gl46.sh 27 test --release --test tier -- --nocapture
#   BLESS=1 gl46.sh 27 test --release --test snapshots -- --nocapture
#   gl46.sh 29 test --release --no-default-features --features wgpu29,scenes,golden,counters --test tier -- --nocapture
#
# The patched copy goes under ~/.cache/vpl-gl46/. The testkit's Cargo.lock is
# restored on exit, so a later plain cargo run uses stock wgpu-hal again.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

leg="$1"
shift
case "$leg" in
    27 | 29 | 30) ;;
    *) echo "leg must be 27, 29 or 30, not '$leg'" >&2; exit 2 ;;
esac

TESTKIT="${TESTKIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/crates/viewport-lib-testkit}"
if [[ ! -f "$TESTKIT/Cargo.toml" ]]; then
    echo "no testkit at $TESTKIT; set TESTKIT=<path to crates/viewport-lib-testkit>" >&2
    exit 1
fi
cd "$TESTKIT"

# Make sure the lockfile and the registry sources exist before reading them.
cargo fetch --quiet

version="$(awk -v leg="$leg" '
    $0 == "name = \"wgpu-hal\"" { getline; gsub(/version = |"/, ""); if (index($0, leg ".") == 1) print }
' Cargo.lock | head -n1)"
if [[ -z "$version" ]]; then
    echo "Cargo.lock has no wgpu-hal $leg.x" >&2
    exit 1
fi

src="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 -type d -name "wgpu-hal-$version" | head -n1)"
if [[ -z "$src" ]]; then
    echo "wgpu-hal-$version is not in the cargo registry" >&2
    exit 1
fi

dst="$HOME/.cache/vpl-gl46/v2/wgpu-hal-$version"
if [[ ! -f "$dst/.gl46-patched" ]]; then
    if [[ -e "$dst" ]]; then
        echo "$dst exists but was not patched (an earlier run stopped part way); delete it and run again" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
    chmod -R u+w "$dst"

    # The desktop GL attributes are the same four pushes in 27, 29 and 30.
    perl -0pi -e '
        $n = s{(gl_context_attributes\.push\(khronos_egl::CONTEXT_MAJOR_VERSION\);\s*gl_context_attributes\.push\()3(\);\s*gl_context_attributes\.push\(khronos_egl::CONTEXT_MINOR_VERSION\);\s*gl_context_attributes\.push\()3(\);)}{${1}4${2}6${3}};
        die "expected one desktop GL 3.3 request, found $n\n" unless $n == 1;
    ' "$dst/src/gles/egl.rs"

    # Report any pipeline layout that needs more texture or sampler slots
    # than the GL backend's fixed tables hold, with every texture binding.
    # `.into()` takes both 27's `&B` and 29/30's `Option<&B>` entries.
    perl -0pi -e '
        $diag = q{
        if num_textures > 16 || num_samplers > 16 {
            eprintln!(
                "[gl46] pipeline layout {:?}: {} textures, {} samplers (the GL backend holds 16 of each)",
                desc.label, num_textures, num_samplers
            );
            for (g, bgl) in desc.bind_group_layouts.iter().enumerate() {
                let bgl: Option<&super::BindGroupLayout> = (*bgl).into();
                let Some(bgl) = bgl else { continue };
                for e in bgl.entries.iter() {
                    if let wgt::BindingType::Texture { sample_type, view_dimension, .. } = e.ty {
                        eprintln!(
                            "[gl46]   group {} binding {}: {:?} {:?}, visible to {:?}",
                            g, e.binding, view_dimension, sample_type, e.visibility
                        );
                    }
                }
            }
        }

};
        $n = s{(\n        self\.counters\.pipeline_layouts\.add\(1\);)}{\n$diag$1};
        die "expected one pipeline layout counter, found $n\n" unless $n == 1;
    ' "$dst/src/gles/device.rs"

    touch "$dst/.gl46-patched"
    echo "patched $dst (desktop GL 3.3 -> 4.6, layout slot report)" >&2
fi

cp Cargo.lock "Cargo.lock.gl46-backup"
trap 'mv -f "$TESTKIT/Cargo.lock.gl46-backup" "$TESTKIT/Cargo.lock"' EXIT

export WGPU_BACKEND=gl
cargo --config "patch.crates-io.wgpu-hal.path='$dst'" "$@"
