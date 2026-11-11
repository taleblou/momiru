// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | BUILD PROVENANCE AND RELEASE ADMISSION
// ==============================================================================
//
// Project:       Momiru
// Component:     Build Infrastructure
// Subcomponent:  Conservative Codegen and Toolchain Admission
// Module:        build_script
// Source:        build.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Build-time collection of compiler/target/profile identifiers and conservative
//   native artifact admission metadata; metadata is NOT a digital signature.
//
// Responsibilities:
//   - Detect explicit incompatible release-profile/codegen environment overrides.
//   - Verify the precise numeric Rust toolchain pin against the compiler version.
//   - Classify standard release output directory shapes without trusting path alone.
//   - Emit MOMIRU_BUILD_* values required by validated CPU native code paths.
//

// ------------------------------------------------------------------------------
// DESIGN AND ENGINEERING GUIDELINES
// ------------------------------------------------------------------------------
//
// Correctness:
//   - Reject unverified code-generation overrides before native admission.
//   - Require both exact toolchain pin and standard release profile metadata.
//   - Never classify compiler flags or filesystem paths as signed provenance.
//   - Retain the release admission tests under tensor_light/build_admission.
//
// Performance Objectives:
//   - Compute policy once at compile time and export constant build metadata.
//   - Never add runtime profiling or dispatch decisions in a build script.
//
// Memory Safety:
//   - Prefer safe Rust APIs that preserve zero-cost ownership and borrowing.
//   - Prohibit unintended aliasing, invalid references and data races.
//   - Restrict unsafe code to documented, reviewed and justified boundaries.
//   - Document soundness invariants for every unsafe operation or block.
//   - Validate bounds, allocation provenance, alignment and initialization.
//   - Preserve correct slice, view, stride and owner lifetime relationships.
//   - Keep device or foreign handles opaque unless validated by real providers.
//   - Require exclusive mutable access where aliasing can occur.
//   - Maintain soundness across public and internal API boundaries.
//
// Maintainability:
//   - Follow idiomatic Rust conventions and consistent module organization.
//   - Keep responsibilities explicit and components independently testable.
//   - Centralize shared shape, indexing, error and layout contracts.
//   - Keep public behavior separate from implementation optimizations.
//   - Document public APIs with Rustdoc and preserve existing module docs.
//   - Maintain public compatibility unless changes are explicitly intended.
//   - Ensure optimized and reference paths have compatible semantics.
//   - Update unit, integration and regression tests for behavioral changes.
//   - Verify performance changes with representative repeatable benchmarks.
//   - Favor measurable specialization over unnecessary duplicated complexity.
//
// ------------------------------------------------------------------------------
// PROJECT INFORMATION
// ------------------------------------------------------------------------------
//
// Authors:
//   Morteza Taleblou
//   Mitra Daneshmand
//
// Organization:  https://github.com/momiru
// Repository:    https://github.com/taleblou/momiru
//
// Maintainer Website:
//   https://taleblou.ir/
//
// Maintainer LinkedIn:
//   https://www.linkedin.com/in/morteza-taleblou
//
// ------------------------------------------------------------------------------
// LICENSING AND ATTRIBUTION
// ------------------------------------------------------------------------------
//
// License:       MIT
// License File:  LICENSE
//
// Copyright and Licensing:
//   This source file is distributed under the MIT License.
//   The SPDX declarations above identify its applicable license and
//   stated copyright holders.
//
//   Redistributions containing this software or substantial portions
//   of it must retain the applicable copyright and permission notices
//   as required by the MIT License.
//
// Attribution:
//   Acknowledgment of the Momiru project in downstream documentation
//   is appreciated but is not an additional licensing requirement.
//
// ==============================================================================

use std::{env, fs, path::Path, process::Command};

/// Reject explicitly supplied code-generation settings diverging from the
/// reviewed release build. An absence of overrides is not proof of authenticity.
pub(crate) fn has_unverified_codegen_override(
    target: &str,
    get: impl Fn(&str) -> Option<String>,
) -> bool {
    const EXACT: &[(&str, &str)] = &[
        ("CARGO_PROFILE_RELEASE_OPT_LEVEL", "3"),
        ("CARGO_PROFILE_RELEASE_LTO", "thin"),
        ("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1"),
        ("CARGO_PROFILE_RELEASE_PANIC", "abort"),
        ("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", "false"),
        ("CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS", "false"),
        ("CARGO_PROFILE_RELEASE_INCREMENTAL", "false"),
        ("CARGO_PROFILE_RELEASE_DEBUG", "0"),
        ("CARGO_PROFILE_RELEASE_STRIP", "none"),
        ("CARGO_PROFILE_RELEASE_RPATH", "false"),
    ];
    for &(key, expected) in EXACT {
        if let Some(value) = get(key) {
            if !value.trim().eq_ignore_ascii_case(expected) { return true; }
        }
    }
    if let Some(value) = get("CARGO_INCREMENTAL") {
        if !matches!(value.trim(), "" | "0" | "false" | "FALSE") { return true; }
    }
    const UNSAFE_OVERRIDES: &[&str] = &[
        "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS",
        "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTDOCFLAGS",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_OPT_LEVEL",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_LTO",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_RUSTFLAGS",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_INCREMENTAL",
        "CARGO_PROFILE_RELEASE_RUSTFLAGS",
        "CARGO_BUILD_TARGET",
    ];
    if UNSAFE_OVERRIDES.iter().any(|key| get(key).is_some_and(|value| !value.is_empty())) {
        return true;
    }
    let normalized = target.to_ascii_uppercase().replace('-', "_").replace('.', "_");
    for suffix in ["RUSTFLAGS", "LINKER", "RUNNER"] {
        let key = format!("CARGO_TARGET_{normalized}_{suffix}");
        if get(&key).is_some_and(|value| !value.is_empty()) { return true; }
    }
    false
}

/// Recognize only Cargo's standard release *path shape*. A matching directory
/// alone does not establish compilation, signer, or artifact provenance.
pub(crate) fn standard_release_output_directory(out: &str) -> bool {
    let path = out.replace('\\', "/");
    let components: Vec<_> = path.split('/').filter(|part| !part.is_empty()).collect();
    if components.iter().any(|part| matches!(*part, "." | "..")) { return false; }
    if components.last() != Some(&"out") { return false; }
    let n = components.len();
    if n < 5 { return false; }
    let is_hex = |s: &str| s.len() >= 8 && s.bytes().all(|c| c.is_ascii_hexdigit());
    if components[n-3] == "build" && components[n-4] == "release" {
        let folder = components[n-2];
        if let Some(hash) = folder.strip_prefix("momiru-") { return is_hex(hash); }
    }
    if n >= 6 && components[n-4] == "build" && components[n-5] == "release"
        && components[n-3] == "momiru"
    {
        return is_hex(components[n-2]);
    }
    false
}

/// Require exactly one exact numeric Rust toolchain channel in the [toolchain]
/// section. A `stable`/nightly alias cannot qualify as a reproducible pin.
pub(crate) fn verified_toolchain_pin(source: Option<&str>, version: &str) -> bool {
    let Some(source) = source else { return false; };
    if !is_numeric_toolchain_version(version) { return false; }
    let mut in_section = false;
    let mut found = None;
    for raw in source.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_section = line == "[toolchain]";
            continue;
        }
        if !in_section || line.is_empty() || line.starts_with('#') { continue; }
        let Some((key, value)) = line.split_once('=') else { continue; };
        if key.trim() != "channel" { continue; }
        if found.is_some() { return false; }
        let value = value.trim();
        let Some(rest) = value.strip_prefix('"') else { return false; };
        let Some((quoted, trailing)) = rest.split_once('"') else { return false; };
        if !trailing.trim().is_empty() && !trailing.trim().starts_with('#') { return false; }
        found = Some(quoted);
    }
    found == Some(version)
}

fn is_numeric_toolchain_version(s: &str) -> bool {
    let chunks: Vec<_> = s.split('.').collect();
    chunks.len() == 3
        && chunks.iter().all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

fn emit(key: &str, value: impl std::fmt::Display) {
    println!("cargo:rustc-env={key}={value}");
}

fn main() {
    for key in ["PROFILE", "OPT_LEVEL", "OUT_DIR", "TARGET", "RUSTC", "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_INCREMENTAL"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    // Explicit rerun directives disable Cargo's broad default env tracking;
    // therefore every inspected override must also be a rerun dependency.
    for key in [
        "CARGO_PROFILE_RELEASE_OPT_LEVEL", "CARGO_PROFILE_RELEASE_LTO",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "CARGO_PROFILE_RELEASE_PANIC",
        "CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS",
        "CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS",
        "CARGO_PROFILE_RELEASE_INCREMENTAL", "CARGO_PROFILE_RELEASE_DEBUG",
        "CARGO_PROFILE_RELEASE_STRIP", "CARGO_PROFILE_RELEASE_RPATH",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_OPT_LEVEL",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_LTO",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_RUSTFLAGS",
        "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_INCREMENTAL",
        "CARGO_PROFILE_RELEASE_RUSTFLAGS", "CARGO_BUILD_TARGET",
        "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTDOCFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=rust-toolchain.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let profile = env::var("PROFILE").unwrap_or_default();
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let normalized_target = target.to_ascii_uppercase().replace('-', "_").replace('.', "_");
    for suffix in ["RUSTFLAGS", "LINKER", "RUNNER"] {
        println!("cargo:rerun-if-env-changed=CARGO_TARGET_{normalized_target}_{suffix}");
    }
    let out_dir = env::var("OUT_DIR").unwrap_or_default();
    let toolchain = env::var("CARGO_MANIFEST_DIR").ok()
        .map(|root| Path::new(&root).join("rust-toolchain.toml"))
        .and_then(|path| fs::read_to_string(path).ok());
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_full = Command::new(rustc).arg("--version").output().ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_else(|| "unknown".to_owned()).trim().to_owned();
    let rustc_version = rustc_full.split_whitespace().nth(1).unwrap_or("");
    let pinned = verified_toolchain_pin(toolchain.as_deref(), rustc_version);
    let overridden = has_unverified_codegen_override(&target, |name| env::var(name).ok());
    let standard = profile == "release"
        && opt_level == "3"
        && standard_release_output_directory(&out_dir);
    emit("MOMIRU_BUILD_PROFILE", &profile);
    emit("MOMIRU_BUILD_OPT_LEVEL", &opt_level);
    emit("MOMIRU_BUILD_STANDARD_RELEASE", standard);
    emit("MOMIRU_BUILD_PINNED", pinned);
    emit("MOMIRU_BUILD_OVERRIDDEN", overridden);
    emit("MOMIRU_BUILD_RUSTC", &rustc_full);
    emit("MOMIRU_BUILD_TARGET", &target);
    if !pinned || overridden {
        println!("cargo:warning=Momiru native release admission stays conservative: unpinned or overridden compilation");
    }
}
