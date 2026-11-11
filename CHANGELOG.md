<!-- SPDX-License-Identifier: MIT -->
# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning as a project convention.

## [Unreleased]

### Documentation
- Expanded the source-backed documentation portal, edition-specific API references, execution-plan trust boundaries, compatibility matrix, troubleshooting, testing/benchmark methodology, and GitHub publication checklist. No Rust implementation, algorithms, or library dependencies were changed.

- Reviewed source-only GitHub distribution to preserve Rust as the sole programming-language implementation (`.rs`). Non-Rust files are licensing, documentation, Cargo/GitHub configuration and required Git folder markers; no Rust algorithms were changed in this packaging audit.

### GitHub Actions hardening
- Replaced mutable `dtolnay/rust-toolchain@master` with direct installation of the repository-pinned Rust `1.99.1` compiler using `rustup`.
- Pinned GitHub Checkout (`v4.4.0`) and upload-artifact (`v4.6.2`) to verified upstream commit SHAs, and disabled persisted checkout credentials in each workflow.
- Replaced the legacy `rustsec/audit-check` workflow action with a pinned `cargo-audit 0.22.2` installation followed by `cargo audit --file Cargo.lock`, retaining scheduled and pull-request security scans. This is a CI configuration change, not a library dependency change. CI execution still requires GitHub-hosted runners and a network connection.

### Changed
- Consolidated the two formerly separate Rust source editions under the single `momiru` crate as `momiru::tensor` and `momiru::tensor_light`.
- Namespaced Cargo test targets, examples and offline benchmarks to avoid collisions.
- Relocated internal unit-test include paths, updated import paths and rebuilt project scaffolding.

### Added
- Root Cargo manifest, locked toolchain, conservative build metadata, CI workflows and developer/security documentation.

### Validation and metadata
- Clarified that the nearest-rank benchmark helper is compiled under each consuming benchmark target, not under an imaginary shared `benches::support` Rust namespace; standardized the new crate/build-script metadata header alignment.
- Reconciled test header `Module` identifiers with the actual Cargo test-harness and library-unit module paths; executable test code is unchanged.
- Removed unnecessary `--allow-dirty` from release packaging, retaining Cargo's clean-source check.

### Migration
- Former imports of `momiru::{Tensor, ...}` from the generic edition are now `momiru::tensor::{Tensor, ...}`.
- Former imports of `momiru::{Tensor, ...}` from the fixed-rank edition are now `momiru::tensor_light::{Tensor, ...}`.

## [0.1.0] - 2026-10-24

- Initial consolidated repository layout. This entry describes packaging; it does not assert that hardware conformance, compilation or benchmarking ran successfully.
