<!-- SPDX-License-Identifier: MIT -->
# Security Policy

## Supported versions

The consolidated project is in an early development stage. No formal security-support SLA is promised. Review the latest repository revision before reporting or deploying the code.

## Vulnerability reporting

**Do not publish exploit details in a public issue.** Prefer a private GitHub vulnerability report through the repository's Security tab (if enabled), or contact the repository maintainer through a private channel listed on their profile. Include affected commit/version, impact, reproduction and conditions. Please avoid sharing credentials, user data or a live exploit payload.

## Threat boundaries

- `tensor` contains generic host tensor/view contracts plus plan-bound typed execution; validated resource lifetimes and aliasing are essential.
- `tensor_light` contains fixed-rank f32 storage, graph/IR and CPU execution; target/device metadata is not a runtime authorization token.
- Native CPU admission metadata from `build.rs` is conservative build context, **not** a signature or independent supply-chain verification.
- Canonical catalogs with `NOT_RUN` cases must not be reported as empirically qualified kernels or platform support.
- Unsafe blocks require explicit invariants and regression coverage.

See [docs/security/safety.md](docs/security/safety.md) for engineering rules.
