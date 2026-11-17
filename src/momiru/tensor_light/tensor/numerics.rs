// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | TENSOR CORE
// ==============================================================================
//
// Project:       Momiru
// Component:     Tensor
// Subcomponent:  Numerics
// Module:        momiru::tensor_light::tensor::numerics
// Source:        src/tensor_light/tensor/numerics.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Pure scalar f32 reference rules for offline checking of canonical numerical
//   contracts.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Specify scalar minimum, maximum and arg-reduction edge semantics.
//   - Provide independent numerical reference behavior for validation.
//   - Keep bulk kernels and runtime planning outside this module.
//
// ------------------------------------------------------------------------------
// DESIGN AND ENGINEERING GUIDELINES
// ------------------------------------------------------------------------------
//
// Correctness:
//   - Keep the concrete tensor element type f32 and fixed ranks 0 through 8.
//   - Validate shapes, strides, broadcasting and dimensional compatibility.
//   - Preserve frozen UASDEA plan semantics and placement contracts.
//   - Distinguish host-side metadata operations from plan-bound numerical work.
//   - Maintain result rank, shape and f32 numerical semantics as documented.
//   - Handle scalar rank zero, empty extents, NaN and boundary conditions.
//   - Check dimension products, byte spans and index arithmetic for overflow.
//   - Reject unsupported operations and invalid resource bindings explicitly.
//   - Make error and failure atomicity contracts clear at public boundaries.
//   - Respect Rust ownership, borrowing and thread-safety requirements.
//
// Performance Objectives:
//   - Prioritize maximum practical runtime performance without changing
//     semantics.
//   - Avoid unnecessary heap allocation, payload copying and materialization.
//   - Favor contiguous access, locality and efficient logical traversal.
//   - Hoist invariant shape, stride and feature checks out of hot loops.
//   - Minimize per-element branching, bounds rechecks and dispatch overhead.
//   - Exploit compile-time specialization or monomorphization when beneficial.
//   - Use SIMD or parallel work only when correctness and measurements justify
//     it.
//   - Keep replay free from avoidable planning, tuning and synchronization.
//   - Prefer optimized paths with safe, equivalent reference fallbacks.
//   - Use reproducible benchmarks instead of unmeasured speed assertions.
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

//! Pure f32 scalar reference semantics for offline numerical validation.
//! These functions are *not* runtime bulk tensor kernels or a hot planner.
//! Exact UASDEA native kernels must conform to these semantics when the
//! default canonical min/max/argmin/argmax numeric contract is selected.

pub const CANONICAL_QUIET_NAN32: f32 = f32::from_bits(0x7fc0_0000);

/// Propagate every NaN with the exact canonical quiet-NaN bit pattern.
/// Distinguish negative zero (lower) from positive zero (higher).
#[inline]
pub fn canonical_minimum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { return CANONICAL_QUIET_NAN32; }
    if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() || b.is_sign_negative() { return -0.0; }
        return 0.0;
    }
    if b < a { b } else { a }
}

/// Canonical maximum (NaN propagates, positive zero is larger than negative).
#[inline]
pub fn canonical_maximum(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { return CANONICAL_QUIET_NAN32; }
    if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() || b.is_sign_positive() { return 0.0; }
        return -0.0;
    }
    if b > a { b } else { a }
}

/// Argmin replacement rule. The first NaN wins if any NaN exists; otherwise
/// the first equal-value position wins, with -0.0 ordered before +0.0.
/// `best` precedes `candidate` in logical C order.
#[inline]
pub fn replace_argmin(best: f32, candidate: f32) -> bool {
    if best.is_nan() { return false; }
    if candidate.is_nan() { return true; }
    if best == 0.0 && candidate == 0.0 {
        return best.is_sign_positive() && candidate.is_sign_negative();
    }
    candidate < best
}

/// Argmax replacement rule with the same first-logical-index tie policy.
#[inline]
pub fn replace_argmax(best: f32, candidate: f32) -> bool {
    if best.is_nan() { return false; }
    if candidate.is_nan() { return true; }
    if best == 0.0 && candidate == 0.0 {
        return best.is_sign_negative() && candidate.is_sign_positive();
    }
    candidate > best
}
