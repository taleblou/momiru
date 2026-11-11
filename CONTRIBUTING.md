<!-- SPDX-License-Identifier: MIT -->
# Contributing to Momiru

Please open a GitHub issue describing behavioral changes before significant API or ABI changes. Keep `tensor` and `tensor_light` conceptually independent; do not move types between them without an explicit compatibility design.

## Local validation

```bash
cargo test --locked
cargo test --locked --all-features
cargo check --locked --all-targets
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Treat formatting and lint enforcement as validation goals, not as proof that existing source was prevalidated. Generate the full public API with `cargo doc --locked --no-deps` when changing public interfaces; update [API reference](docs/reference/README.md), [compatibility](docs/compatibility.md), and [release checklist](docs/development/release.md) only where the source warrants it. For CPU performance changes, run the matching `benches/tensor_*.rs` or `benches/tensor_light_*.rs` target and record toolchain, OS, CPU, baseline, method and noise. Do not introduce runtime performance search into frozen-plan replay.

## Pull request expectations

Include the motivation, API compatibility assessment, test/regression coverage, unsafe-code soundness review (if applicable), benchmark design and documentation changes. Keep public Rustdoc and standardized `Source:`/`Module:` header fields synchronized with paths. Use the SPDX MIT header in new Rust code.

## Security

For privately exploitable reports, use the procedure in [SECURITY.md](SECURITY.md) rather than a public issue.
