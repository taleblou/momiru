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
// Subcomponent:  Axis
// Module:        momiru::tensor_light::tensor::axis
// Source:        src/tensor_light/tensor/axis.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Rank-checked reduction-axis descriptors and output-rank metadata for
//   canonical UASDEA IR.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Validate reduction axis selectors against compile-time input ranks.
//   - Describe resulting axis metadata and canonical operation identity.
//   - Keep scalar reduction computation outside this module.
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

//! Pure rank-checked reduction-axis metadata for the canonical UASDEA IR.
//! No numeric reductions execute in this module. The exact axis selector and
//! output rank are included in a separately frozen graph/plan identity.
use super::{Rank, TensorError, ValidRank};

/// `All` and `Explicit([])` are intentionally *different* contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionAxes<const R: usize>
where
    Rank<R>: ValidRank,
{
    All,
    Explicit([bool; R]),
}

impl<const R: usize> ReductionAxes<R>
where
    Rank<R>: ValidRank,
{
    #[inline]
    pub const fn all() -> Self { Self::All }

    pub fn explicit(axes: &[usize]) -> Result<Self, TensorError> {
        let mut selected = [false; R];
        if axes.len() > R { return Err(TensorError::InvalidReductionAxes); }
        for &axis in axes {
            if axis >= R || selected[axis] { return Err(TensorError::InvalidReductionAxes); }
            selected[axis] = true;
        }
        Ok(Self::Explicit(selected))
    }

    pub fn is_explicit_empty(&self) -> bool {
        matches!(self, Self::Explicit(selected) if !selected.contains(&true))
    }

    #[inline]
    pub fn selects(&self, axis: usize) -> bool {
        if axis >= R { return false; }
        match self {
            Self::All => true,
            Self::Explicit(selected) => selected[axis],
        }
    }

    pub fn selected_count(&self) -> usize {
        match self {
            Self::All => R,
            Self::Explicit(axes) => axes.iter().filter(|&&selected| selected).count(),
        }
    }

    /// Arg-reduction index ABI supports exactly one explicitly selected axis,
    /// or `All` (the single flattened C-order index). A multi-axis explicit
    /// selector cannot be encoded as one axis-relative result, and must be
    /// rejected *here*, matching the canonical IR reduction validator.
    /// `All` on a rank-zero scalar is valid and yields index zero.
    pub fn validate_arg_reduction(&self) -> Result<(), TensorError> {
        match self {
            Self::All => Ok(()),
            Self::Explicit(_) if self.selected_count() == 1 => Ok(()),
            Self::Explicit(_) => Err(TensorError::InvalidReductionAxes),
        }
    }

    /// Validate keepdims and the caller-chosen compile-time output rank.
    pub fn output_dims<const OUT: usize>(
        &self, input: [usize; R], keepdims: bool,
    ) -> Result<[usize; OUT], TensorError>
    where
        Rank<OUT>: ValidRank,
    {
        let expected = if keepdims { R } else { R - self.selected_count() };
        if OUT != expected { return Err(TensorError::InvalidReductionAxes); }
        let mut dims = [0usize; OUT];
        let mut dst = 0usize;
        for (src, dim) in input.iter().copied().enumerate() {
            if self.selects(src) {
                if keepdims {
                    dims[dst] = 1;
                    dst += 1;
                }
            } else {
                dims[dst] = dim;
                dst += 1;
            }
        }
        debug_assert_eq!(dst, OUT);
        Ok(dims)
    }

    /// True exactly when at least one output cell exists and its contracted
    /// input domain has cardinality zero. An output with zero cells is *not*
    /// an evaluated empty reduction slice.
    pub fn evaluates_empty_slice<const OUT: usize>(
        &self, input: [usize; R], keepdims: bool,
    ) -> Result<bool, TensorError>
    where
        Rank<OUT>: ValidRank,
    {
        let output = self.output_dims::<OUT>(input, keepdims)?;
        if output.contains(&0) { return Ok(false); }
        if self.is_explicit_empty() { return Ok(false); }
        for (axis, extent) in input.iter().enumerate() {
            if self.selects(axis) && *extent == 0 { return Ok(true); }
        }
        Ok(false)
    }
}
