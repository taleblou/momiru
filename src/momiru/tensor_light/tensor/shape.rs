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
// Subcomponent:  Shape
// Module:        momiru::tensor_light::tensor::shape
// Source:        src/tensor_light/tensor/shape.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Fixed-rank tensor shape, stride and dimension-product metadata.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Validate extents and compile-time ranks from 0 through 8.
//   - Compute safe shape and stride metadata with checked arithmetic.
//   - Support scalar, empty and contiguous shape cases.
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

//! Dedicated fixed-rank tensor shape module.
//! Fixed-rank tensor shape and stride metadata.

use super::{Rank, TensorError, ValidRank};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape<const R: usize>
where
    Rank<R>: ValidRank,
{
    dims: [usize; R],
    strides: [usize; R],
    len: usize,
    span_len: usize,
}

impl<const R: usize> Shape<R>
where
    Rank<R>: ValidRank,
{
    pub fn contiguous(dims: [usize; R]) -> Result<Self, TensorError> {
        // Empty shapes are valid even when products of other extents overflow.
        // Their logical access set is empty; zero strides are canonical here.
        if dims.contains(&0) {
            return Ok(Self { dims, strides: [0usize; R], len: 0, span_len: 0 });
        }
        let mut strides = [0usize; R];
        let mut stride = 1usize;
        let mut axis = R;
        while axis > 0 {
            axis -= 1;
            strides[axis] = stride;
            stride = stride
                .checked_mul(dims[axis])
                .ok_or(TensorError::ShapeOverflow)?;
        }

        stride.checked_mul(core::mem::size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(TensorError::AllocationOverflow)?;
        Ok(Self { dims, strides, len: stride, span_len: stride })
    }

    pub fn strided(dims: [usize; R], strides: [usize; R]) -> Result<Self, TensorError> {
        // The canonical UASDEA signed-stride ABI uses isize. Even an empty
        // view retains representable structural stride metadata.
        if strides.iter().any(|&stride| stride > isize::MAX as usize) {
            return Err(TensorError::ShapeOverflow);
        }
        // An empty mapping must not multiply any extent by a stride.
        if dims.contains(&0) {
            return Ok(Self { dims, strides, len: 0, span_len: 0 });
        }
        let mut len = 1usize;
        for &dim in &dims {
            len = len.checked_mul(dim).ok_or(TensorError::ShapeOverflow)?;
        }
        let span_len = if R == 0 {
            1
        } else {
            let mut max_offset = 0usize;
            let mut i = 0usize;
            while i < R {
                let extent = dims[i]
                    .saturating_sub(1)
                    .checked_mul(strides[i])
                    .ok_or(TensorError::ShapeOverflow)?;
                max_offset = max_offset
                    .checked_add(extent)
                    .ok_or(TensorError::ShapeOverflow)?;
                i += 1;
            }
            max_offset
                .checked_add(1)
                .ok_or(TensorError::ShapeOverflow)?
        };

        for count in [len, span_len] {
            count.checked_mul(core::mem::size_of::<f32>())
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or(TensorError::AllocationOverflow)?;
        }
        Ok(Self { dims, strides, len, span_len })
    }

    #[inline(always)]
    pub const fn rank(&self) -> usize {
        R
    }

    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] {
        &self.dims
    }

    #[inline(always)]
    pub const fn strides(&self) -> &[usize; R] {
        &self.strides
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Logical element cardinality (rank zero is one, any zero extent is zero).
    /// Uses previously validated shape metadata; it never recomputes a product.
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.len }

    #[inline(always)]
    pub const fn span_len(&self) -> usize {
        self.span_len
    }

    pub fn is_contiguous(&self) -> bool {
        if self.len == 0 {
            return true;
        }

        let mut expected = 1usize;
        let mut axis = R;
        while axis > 0 {
            axis -= 1;
            if self.dims[axis] > 1 && self.strides[axis] != expected {
                return false;
            }
            expected *= self.dims[axis];
        }
        true
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.len == 0 }

    #[inline(always)]
    pub const fn is_scalar(&self) -> bool { R == 0 }

    /// Structural signed-stride metadata for the canonical graph ABI.
    /// All positive strides have already been verified to fit `isize`.
    pub fn signed_strides(&self) -> [isize; R] {
        core::array::from_fn(|i| self.strides[i] as isize)
    }

    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }

    /// Logical payload bytes for the canonical f32 Tensor/Shape. This
    /// metadata is also reused for auxiliary index/mask *geometry*, but
    /// its byte count is ALWAYS f32-sized. For actual auxiliary payload
    /// bytes use `IndexResult::nbytes()` or `PredicateMask::nbytes()`.
    #[inline(always)]
    pub const fn nbytes(&self) -> usize { self.len * core::mem::size_of::<f32>() }

    #[inline(always)]
    pub fn is_c_contiguous(&self) -> bool { self.is_contiguous() }

    pub fn is_f_contiguous(&self) -> bool {
        if self.len == 0 { return true; }
        let mut expected = 1usize;
        for axis in 0..R {
            if self.dims[axis] > 1 && self.strides[axis] != expected { return false; }
            expected *= self.dims[axis];
        }
        true
    }

    #[inline]
    pub fn internal_overlap(&self) -> super::InternalOverlap {
        // One shared tri-state certificate: an exact multi-axis collision is
        // Definite, not Unknown. The common separated-stride case stays fast.
        super::signed_mapping_overlap(&self.dims, &self.signed_strides(), self.len)
    }

    #[inline]
    pub fn is_unique(&self) -> bool {
        self.internal_overlap() == super::InternalOverlap::None
    }

    #[inline]
    pub fn is_exhaustive(&self) -> bool {
        self.is_unique() && self.len == self.span_len
    }

    /// Conservative canonical non-overlap test for non-negative-stride mutable views.
    /// Common dense, transposed, narrowed, and gapped layouts are accepted;
    /// inconclusive small (<=128-element) mappings are checked exactly.
    /// Larger unproven layouts remain rejected rather than becoming unsafe.
    pub(crate) fn is_non_overlapping_for_mutation(&self) -> bool {
        // All Shape strides are representable isize values by construction.
        // The same sound certificate is used by mutable signed views.
        super::signed_mapping_is_injective(
            &self.dims, &self.signed_strides(), self.len,
        )
    }
}
