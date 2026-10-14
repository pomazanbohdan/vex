# Changelog

All notable changes to vex are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Add Dart indexing with the pinned Orchard grammar and an optional SDK Analysis Server backend for strict references. The Dart client validates source identity, UTF-16 offsets and complete search streams.

- **`install.sh`**: `curl -fsSL https://raw.githubusercontent.com/tenatarika/vex/main/install.sh | sh`
  installs `vex` and `vex-mcp` from the latest release into `~/.local/bin`
  (macOS arm64, Linux x86_64; `VEX_VERSION`, `VEX_INSTALL_DIR`, `VEX_NO_MCP`).
  It refuses musl and glibc older than 2.34 with a pointer to a source build.
- **`cargo binstall vex-search` / `vex-search-mcp`** download the prebuilt
  release binaries instead of compiling.


### Fixed

- Keep the Dart Analysis Server cache outside application directories when Windows cache environment values are empty.
- Include Dart `main` test entrypoints in `tests-for` after matching the test path; keep native and utility name filters.
- Return empty strict results without an SDK when no exact Dart definition matches, while retaining source freshness checks and omitting compiler provenance.
