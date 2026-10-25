#!/usr/bin/env bash
# Time the overlay arena's UI scenes on Mesa's CPU renderers in an arm64 Linux
# container: llvmpipe through wgpu's GL backend, and lavapipe through Vulkan.
#
# This estimates whether a small board with no usable GPU (or one stuck at
# GL ES 2.0) could draw viewport-lib-ui screens through the existing renderer,
# with no new backend. The container runs on the host's own cores, so the
# numbers are an upper bound for a Cortex-A53 board such as a Raspberry Pi 3 or
# Zero 2 W: several times faster per core. --cpus defaults to 4 to match those
# boards' core count. The arena's target is 1280x800, larger than a typical
# 800x480 kiosk panel.
#
# Usage:
#   scripts/llvmpipe_arena.sh [--backend gl|vulkan|both] [--cpus N] [--shell]
#
#   --backend  which CPU renderer to time (default both)
#   --cpus     cores the container is pinned to, and llvmpipe's thread count
#              (default 4)
#   --shell    open a shell in the container with Mesa installed and the GL
#              environment set, instead of running anything
#
# Needs Docker with arm64 Linux support (Docker Desktop or OrbStack on Apple
# silicon run it natively). Path dependencies reach outside the vpl folder
# (viewport-lib-ui's taffy is a clone under ../../oss), so the directory two
# levels above the vpl folder is mounted at its own absolute path and every
# relative path resolves as it does on the host. VPL_MOUNT_ROOT overrides it.
# The build lives in a Docker volume, not the host's target directory, because
# it is a different target triple. Results land in the arena's runs/ folder as
# runs/cpu-<backend>-<stamp>/suite/.
set -euo pipefail

backend="both"
cpus=4
shell=0
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --shell) shell=1; shift ;;
        -h|--help) sed -n 2,27p "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
case "$backend" in
    gl|vulkan|both) ;;
    *) echo "--backend takes gl, vulkan or both" >&2; exit 1 ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
vpl_root="$(cd "$repo_root/.." && pwd)"
mount_root="${VPL_MOUNT_ROOT:-$(cd "$vpl_root/../.." && pwd)}"
arena="$vpl_root/harness/viewport-lib-overlay-arena"
if [ ! -d "$arena" ]; then
    echo "no overlay arena at $arena" >&2
    exit 1
fi

# Runs inside the container.
inner=$(cat <<'INNER'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "== installing Mesa (llvmpipe and lavapipe)"
apt-get update -qq
apt-get install -y -qq libegl1 libgl1-mesa-dri libgles2 mesa-utils \
    mesa-vulkan-drivers vulkan-tools python3 >/dev/null

# The GL run hides every Vulkan driver, so wgpu finds only the GL adapter.
# wgpu 27's default instance ignores WGPU_BACKEND, so the drivers present are
# what decide the backend.
gl_env() {
    export LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless WGPU_BACKEND=gl
    export VK_ICD_FILENAMES=/nonexistent.json VK_DRIVER_FILES=/nonexistent.json
}
vulkan_env() {
    unset LIBGL_ALWAYS_SOFTWARE EGL_PLATFORM VK_ICD_FILENAMES VK_DRIVER_FILES
    export WGPU_BACKEND=vulkan
    local icd
    icd="$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -1)"
    if [ -n "$icd" ]; then
        export VK_ICD_FILENAMES="$icd" VK_DRIVER_FILES="$icd"
    fi
}

echo "== CPU: $(nproc) cores visible, llvmpipe threads ${LP_NUM_THREADS:-unset}"
(gl_env; echo "== GL:"; eglinfo -B 2>/dev/null | grep -iE "renderer|version string" | head -4 || true)
(vulkan_env; echo "== Vulkan:"; vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|apiVersion" | head -2 || true)

if [ "$SHELL_ONLY" = 1 ]; then
    gl_env
    echo "== GL environment set; cd $VPL_ROOT/harness/viewport-lib-overlay-arena to run the arena"
    exec bash
fi

cd "$VPL_ROOT/harness/viewport-lib-overlay-arena"
echo "== building the arena (first run compiles the workspace for arm64 Linux)"
cargo build --release --bins

stamp="$(date +%Y%m%d-%H%M%S)"
for b in $BACKENDS; do
    (
        if [ "$b" = gl ]; then gl_env; else vulkan_env; fi
        export ARENA_PARENT="$VPL_ROOT/harness/viewport-lib-overlay-arena/runs/cpu-$b-$stamp"
        mkdir -p "$ARENA_PARENT"
        echo "== suite on $b -> runs/cpu-$b-$stamp/suite"
        ./scripts/suite.sh --no-settle
        used="$(grep -ho '"gpu"[^,]*\|"backend"[^,]*' "$ARENA_PARENT/suite/suite.json" 2>/dev/null | head -2 | tr '\n' ' ' || true)"
        echo "== $b run used: ${used:-no adapter recorded in suite.json}"
    )
done
INNER
)

backends="$backend"
[ "$backend" = both ] && backends="gl vulkan"

tty_flags=(-i)
[ -t 0 ] && tty_flags=(-it)

exec docker run --rm "${tty_flags[@]}" \
    --platform linux/arm64 \
    --cpuset-cpus "0-$((cpus - 1))" \
    -e LP_NUM_THREADS="$cpus" \
    -v "$mount_root:$mount_root" \
    -v vpl-arm64-cargo-registry:/usr/local/cargo/registry \
    -v vpl-arm64-target:/target \
    -e CARGO_TARGET_DIR=/target \
    -e VPL_ROOT="$vpl_root" \
    -e BACKENDS="$backends" \
    -e SHELL_ONLY="$shell" \
    rust:latest \
    bash -c "$inner"
