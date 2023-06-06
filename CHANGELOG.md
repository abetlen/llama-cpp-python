# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (llama.cpp) k-quants support
- (server) mirostat sampling parameters to server

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