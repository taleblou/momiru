// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | PUBLIC API
// ==============================================================================
//
// Project:       Momiru
// Component:     Tensor Light
// Subcomponent:  Public API and Module Exports
// Module:        momiru::tensor_light
// Source:        src/tensor_light/mod.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Public crate interface for compile-time-rank f32 host tensors and immutable
//   UASDEA frozen execution plans.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Expose the canonical Tensor<const R: usize> API with ranks from 0 to 8.
//   - Organize tensor, plan, runtime and optional offline specialization modules.
//   - Preserve host metadata functionality without adding plan-free arithmetic.
//   - Maintain documented public re-exports and feature boundaries.
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

//! Momiru: fixed-rank f32 tensors and a single UASDEA frozen-plan contract.
//!
//! `Tensor<const R: usize>` is the canonical public host tensor entrypoint.
//! R is compile-time restricted to 0..=8 and element type is exactly f32.
//! Constructors, storage, indexing and zero-copy views are inherent methods.
//! Numerical graph evaluation requires one offline-frozen static dataflow plan.
//! A restricted, prebound built-in f32 CPU provider ships for validated plans;
//! there is no signed disk catalog loader or GPU artifact compiler/backend yet.
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod execution;
pub mod tensor;
#[path = "execution/interop.rs"]
pub mod interop;

/// Canonical externally-facing f32 tensor type, ranks 0..=8.
///
/// Rank nine cannot be instantiated because `Rank<9>` is not a `ValidRank`:
/// ```compile_fail
/// use momiru::tensor_light::Tensor;
/// let _ = Tensor::<9>::try_from_vec([1; 9], vec![1.0f32]);
/// ```
///
/// A general tensor of `f64` is intentionally not part of the public API:
/// ```compile_fail
/// use momiru::tensor_light::tensor::Tensor;
/// let _ = Tensor::<f64, 1>::try_from_vec([1], vec![1.0f64]);
/// ```
pub type Tensor<const R: usize> = tensor::Tensor<f32, R>;
/// Explicit host alias for code distinguishing host from future device tensors.
pub type HostTensor<const R: usize> = Tensor<R>;
/// Opaque uninitialized, owned f32 host capacity; never readable until initialized.
pub type UninitTensor<const R: usize> = tensor::UninitTensor<f32, R>;
pub type TensorView<'a, const R: usize> = tensor::TensorView<'a, f32, R>;
pub type TensorViewMut<'a, const R: usize> = tensor::TensorViewMut<'a, f32, R>;
pub use tensor::{DType, InternalOverlap, Layout, Rank, Shape, SignedSliceSpec, SignedTensorView,
    SignedTensorViewMut, ReductionAxes, IndexResult, PredicateMask, canonical_minimum, canonical_maximum,
    replace_argmin, replace_argmax, CANONICAL_QUIET_NAN32, SliceSpec,
    TensorError, ValidRank, DEFAULT_HOST_ALIGNMENT, MAX_RANK, MIN_RANK};

/// Optional, offline-only UASDEA candidate specialization.
///
/// Enable with `cargo build --features aot`. The default runtime build
/// does not compile this module or perform autotuning.
#[cfg(feature = "aot")]
pub mod aot;
