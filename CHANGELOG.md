# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Changed

- Moved `wireframe_mode` into `ViewportFrame` so viewport display state lives entirely under the grouped viewport section.

## [0.2.0]

### Changed

- Refactored the frame submission API around grouped frame sections.
- Introduced `RenderCamera` as the canonical renderer-facing camera type.
- Moved scene, viewport, interaction, effects, and cache-hint state into dedicated frame sub-objects.
- Updated examples to use the grouped frame API.

## [0.1.0]

### Added

- Initial crates.io release of `viewport-lib`.
