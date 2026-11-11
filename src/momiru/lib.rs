// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | PUBLIC API AND EDITION REGISTRY
// ==============================================================================
//
// Project:       Momiru
// Component:     Crate Interface
// Subcomponent:  Independent Tensor Module Registry
// Module:        momiru
// Source:        src/lib.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Single crate root providing two statically compiled public tensor families.
//   No implicit root-level Tensor alias, eager arithmetic or cross-edition cast.
//
// Responsibilities:
//   - Register momiru::tensor for generic-element runtime-ND storage.
//   - Register momiru::tensor_light for fixed-rank f32 tensors (ranks 0..8).
//   - Prevent public name collisions and accidental ABI/plan interchange.
//

// ------------------------------------------------------------------------------
// DESIGN AND ENGINEERING GUIDELINES
// ------------------------------------------------------------------------------
//
// Correctness:
//   - Keep tensor and tensor_light types and execution plans independent.
//   - Require edition-qualified imports and explicit capability validation.
//   - Preserve f32/f64 dtype contracts, shape checking and execution boundaries.
//   - Do not introduce undisclosed conversion or shared mutable global state.
//
// Performance Objectives:
//   - Static Rust modules add no virtual dispatch, allocation or hot-path wrapper.
//   - Delegate optimizations and vectorization to their owning tensor edition.
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

#![deny(unsafe_op_in_unsafe_fn)]

/// Generic-element, runtime-ND tensor edition.
pub mod tensor;
/// Fixed-rank (0..=8), f32 tensor edition with frozen UASDEA plan execution.
pub mod tensor_light;
