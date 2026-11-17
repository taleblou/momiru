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
// Subcomponent:  Error
// Module:        momiru::tensor_light::tensor::error
// Source:        src/tensor_light/tensor/error.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Dedicated errors for fixed-rank f32 tensor construction, shape checks and
//   host access.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Define stable tensor error variants and diagnostics.
//   - Represent invalid shapes, ranks, indices and allocation boundaries.
//   - Keep error propagation explicit across public tensor APIs.
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

//! Dedicated fixed-rank tensor error module.
//! Tensor API errors.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorError {
    ShapeOverflow,
    AllocationOverflow,
    AllocationFailed,
    ResourceInsufficient,
    LengthMismatch,
    ShapeMismatch,
    InvalidAlignment,
    IndexOutOfBounds,
    InvalidSliceStep,
    InvalidPermutation,
    AxisOutOfBounds,
    InvalidRange,
    InvalidProbability,
    OverlappingMutableView,
    NonContiguous,
    InvalidReductionAxes,
    EmptyReduction,
    InvalidMaskValue,
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::ShapeOverflow => "tensor shape or stride arithmetic overflowed",
            Self::AllocationOverflow => "tensor allocation size overflowed",
            Self::AllocationFailed => "the host allocator could not allocate the requested memory",
            Self::ResourceInsufficient => "the requested tensor exceeds the caller-selected element budget",
            Self::LengthMismatch => "backing storage length does not match the tensor/view span",
            Self::ShapeMismatch => "tensor shapes are incompatible for this operation",
            Self::InvalidAlignment => "alignment must be a nonzero power of two",
            Self::IndexOutOfBounds => "tensor index is out of bounds",
            Self::InvalidSliceStep => "slice step is invalid for this view (zero is never allowed)",
            Self::InvalidPermutation => "axis permutation is invalid",
            Self::AxisOutOfBounds => "tensor axis is out of bounds",
            Self::InvalidRange => "range parameters are invalid",
            Self::InvalidProbability => "probability must be finite and within [0, 1]",
            Self::OverlappingMutableView => "mutable strided view must not map multiple logical indices to the same element",
            Self::NonContiguous => "the logical tensor view is not forward C-contiguous",
            Self::InvalidReductionAxes => "reduction axis selection or output rank is invalid",
            Self::EmptyReduction => "the evaluated reduction domain is empty",
            Self::InvalidMaskValue => "unpacked predicate mask contains a byte other than 0 or 1",
        };
        f.write_str(message)
    }
}

impl std::error::Error for TensorError {}
