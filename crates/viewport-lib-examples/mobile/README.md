# Mobile example

A viewport-lib scene on Android and iOS: winit for the window and the touch events, wgpu for the device, and the built-in `OrbitCameraController` for the camera. Three primitives on a grid, orbited with your fingers.

| Gesture | Action |
|---------|--------|
| 1-finger drag | Orbit |
| 2-finger drag | Pan |
| Pinch | Zoom |
| 2-finger rotate | Roll (iOS only) |

`src/app.rs` holds the whole example and runs unchanged on both platforms. Two things vary by target: Android asks for the Vulkan backend so the device does not pick GLES instead, and iOS gets pinch and rotation as their own winit events, while on Android the pinch span is worked out from the raw touch positions.

The crate sits outside the parent workspace. cargo-mobile2's generated projects build the Rust side by running cargo from this directory and looking for the library under `./target`, and a workspace member writes to the workspace root's target directory instead.

## Desktop

No device needed to work on the scene or the camera:

```sh
cargo mobile-example
```

That alias expands to `cargo run --release --manifest-path crates/viewport-lib-examples/mobile/Cargo.toml --bin mobile-desktop`. Mouse drags reach the same orbit controller the touch handler feeds.

## Prerequisites

- Rust via [rustup](https://rustup.rs), not Homebrew Rust: see Troubleshooting.
- cargo-mobile2: `cargo install cargo-mobile2`
- For iOS: macOS with Xcode and a downloaded simulator.
- For Android: Android Studio, an SDK, and an NDK.

`rust-toolchain.toml` in this directory lists the mobile targets, so rustup installs them the first time you run cargo here.

## First-time setup

`gen/` holds the Xcode and Gradle projects, and it is gitignored: it is generated output, it runs to well over a gigabyte once Gradle has built once, and the Xcode project bakes in absolute paths and an Apple development team id. Generate it yourself:

```sh
cd crates/viewport-lib-examples/mobile
cargo mobile init
```

`cargo mobile init` rewrites `Cargo.toml` and `src/lib.rs` with its own templates. The example lives in `src/app.rs`, which it leaves alone, so restoring is two files:

```sh
git checkout Cargo.toml src/lib.rs
```

Set your own Apple development team id in `mobile.toml` before building for iOS. Xcode reports it under Settings -> Accounts -> your account -> Manage Certificates, and `security find-identity -p codesigning -v` prints it too. Re-run `cargo mobile init` after changing it, since the value is copied into the generated project.

Copy `.cargo/config.toml.example` to `.cargo/config.toml`, on either platform. The real file is gitignored because the NDK linker paths in it are per-machine, but its `[build] target-dir` matters for both: the generated projects run cargo from this directory and then look for the library under `./target`, so a `build.target-dir` set in your `~/.cargo/config.toml` would put it somewhere they do not look. A config in this directory outranks your home one. For Android, point the linkers at your NDK as well.

## Building and running

```sh
cargo apple run      # iOS device or simulator
cargo android run    # connected Android device or emulator
```

Or open the generated projects and drive them from the IDE:

```sh
cargo apple open     # Xcode
cargo android open   # Android Studio
```

In Xcode, pick a simulator or device from the device picker and press Run. Xcode calls cargo to build the Rust library, links it, and boots the simulator.

To compile-check both targets without a device, an NDK, or Xcode:

```sh
scripts/check_mobile.sh
```

## wgpu leg

The example owns its wgpu surface and calls `Surface::get_current_texture`, which returns a `Result` on wgpu 27 and a `CurrentSurfaceTexture` enum from 29 on. It is written against 27 and the manifest bakes viewport-lib's `wgpu27` feature, so unlike the other example crates there is no leg to pick.

## Troubleshooting

**`aarch64-apple-ios-sim` or `aarch64-linux-android` target not found**

If your shell resolves `cargo` to a Homebrew-managed binary, the mobile targets are missing and `rust-toolchain.toml` is ignored, because Homebrew manages its own Rust installation separately from rustup and its cargo is not a rustup proxy. Invoke the rustup cargo directly, or set the `CARGO` environment variable so Xcode's build script picks it up:

```sh
~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo check --target aarch64-apple-ios-sim
```

`scripts/check_mobile.sh` does this for you, and takes `VPL_MOBILE_TOOLCHAIN` if your toolchain directory differs.

**Xcode reports missing symbols after `cargo mobile init`**

init overwrote `Cargo.toml` with its template, which pins a different winit. Run `git checkout Cargo.toml src/lib.rs` and build again.

**The app shows a black screen after returning from the background**

Backgrounding invalidates the surface, so `ApplicationHandler::suspended` drops the device, the renderer and the surface, and `resumed` rebuilds them. Any state you add to `AppState` is rebuilt on that path too; state that should survive belongs outside it.
