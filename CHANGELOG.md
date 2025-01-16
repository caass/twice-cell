# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/caass/twice-cell/compare/v0.1.0...v0.1.1) - 2025-01-16

### Fixed

- *(serde)* fix serde impl for twice-cell
- fix!(miri): Use raw pointers to pass miri checks
- *(err)* Bypass panic hook on errors during initialization
- *(panic)* Bypass panic hook on caught panics during initialization

### Other

- *(ci)* i hate ci (#16)
- *(deps)* update dependencies and fix ci (#15)
- *(readme)* Update README (#14)
- *(CI)* Run MIRI, use more toolchains, etc. (#12)
- *(clippy)* Add safety comments
- Add struct docs to `TwiceCell`
- release

## [0.1.0](https://github.com/caass/twice-cell/releases/tag/v0.1.0) - 2024-09-13

### Added

- implement `TwiceLock`
- Implement `TwiceCell`

### Fixed

- *(TwiceLock)* Stop `TwiceLock::initialize` from propagating panics

### Other

- add CI
- Add description, repo, website
- Add readme and licenses
