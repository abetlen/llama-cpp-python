# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.61]

### Fixed

- Fix broken pip installation

## [0.1.60]

### NOTE

- This release was deleted due to a bug  with the packaging system that caused pip installations to fail.

### Fixed

- Truncate max_tokens in create_completion so requested tokens doesn't exceed context size.
- Temporarily disable cache for completion requests

## [v0.1.59]

### Added

- (llama.cpp) k-quants support
- (server) mirostat sampling parameters to server

### Fixed

- Support both `.so` and `.dylib` for `libllama` on MacOS

## [v0.1.58]

### Added

- (llama.cpp) Metal Silicon support

## [v0.1.57]

### Added

- (llama.cpp) OpenLlama 3B support

## [v0.1.56]

### Added

- (misc) Added first version of the changelog
- (server) Use async routes
- (python-api) Use numpy for internal buffers to reduce memory usage and improve performance.

### Fixed

- (python-api) Performance bug in stop sequence check slowing down streaming.