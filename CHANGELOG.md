# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Added: k-quants support

## [v0.1.58]

- Added: Metal Silicon support

## [v0.1.57]

- Added: OpenLlama 3B support

## [v0.1.56]

### Added

- Added first version of the changelog
- Server: Use async routes
- Use numpy for internal buffers to reduce memory usage and improve performance.

### Fixed

- Performance bug in stop sequence check slowing down streaming.