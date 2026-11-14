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
// Subcomponent:  Layout
// Module:        momiru::tensor_light::tensor::layout
// Source:        src/tensor_light/tensor/layout.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Immutable signed-stride host layout metadata over a trusted allocation.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Validate logical address ranges against known backing storage.
//   - Represent signed strides and offsets without materializing tensor values.
//   - Avoid treating metadata checks as authentication of foreign device
//     pointers.
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

//! Canonical, immutable signed-stride host layout descriptor.
//!
//! A layout is metadata, not an execution schedule or a device pointer.  It
//! validates an f32 logical index set against one *trusted* backing allocation;
//! it cannot authenticate an arbitrary foreign pointer or GPU address.

use super::{InternalOverlap, Rank, Shape, TensorError, ValidRank};

/// Fixed rank (0..=8), signed element strides and signed origin offset.
///
/// Empty tensors have no reachable addresses: `reachable_bounds` is `None`,
/// their element count and reachable span are zero, and an origin one-past
/// the allocation is legal as *metadata only*. Rank zero is a single scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout<const R: usize>
where
    Rank<R>: ValidRank,
{
    dims: [usize; R],
    strides: [isize; R],
    origin: isize,
    len: usize,
    capacity: usize,
    low: Option<usize>,
    high: Option<usize>,
}

impl<const R: usize> Layout<R>
where
    Rank<R>: ValidRank,
{
    /// Validate the complete reachable range; strides and origin are counted
    /// in f32 *elements*, not bytes. No pointer is formed by this function.
    pub fn new(
        dims: [usize; R],
        strides: [isize; R],
        origin: isize,
        capacity: usize,
    ) -> Result<Self, TensorError> {
        // An actual Rust f32 slice cannot represent more than isize::MAX bytes.
        // Device-only (opaque) allocations require a separate target-native ABI.
        let max_elements = (isize::MAX as usize) / core::mem::size_of::<f32>();
        if capacity > max_elements {
            return Err(TensorError::AllocationOverflow);
        }
        if dims.contains(&0) {
            if origin < 0 || usize::try_from(origin).ok().is_none_or(|o| o > capacity) {
                return Err(TensorError::IndexOutOfBounds);
            }
            return Ok(Self {
                dims, strides, origin, len: 0, capacity, low: None, high: None,
            });
        }

        let mut len = 1usize;
        for dim in dims {
            len = len.checked_mul(dim).ok_or(TensorError::ShapeOverflow)?;
        }
        if len > max_elements {
            return Err(TensorError::AllocationOverflow);
        }

        let mut low = origin as i128;
        let mut high = origin as i128;
        for axis in 0..R {
            let extent = i128::try_from(dims[axis] - 1)
                .map_err(|_| TensorError::ShapeOverflow)?;
            let delta = extent
                .checked_mul(strides[axis] as i128)
                .ok_or(TensorError::ShapeOverflow)?;
            if delta < 0 {
                low = low.checked_add(delta).ok_or(TensorError::ShapeOverflow)?;
            } else {
                high = high.checked_add(delta).ok_or(TensorError::ShapeOverflow)?;
            }
        }
        if low < 0 || high >= capacity as i128 {
            return Err(TensorError::IndexOutOfBounds);
        }
        let low = usize::try_from(low).map_err(|_| TensorError::IndexOutOfBounds)?;
        let high = usize::try_from(high).map_err(|_| TensorError::IndexOutOfBounds)?;
        Ok(Self {
            dims, strides, origin, len, capacity, low: Some(low), high: Some(high),
        })
    }

    /// Checked conversion from the legacy nonnegative-stride `Shape` metadata.
    /// New signed-stride operations should use `Layout` as their descriptor.
    pub fn from_shape(shape: Shape<R>, origin: isize, capacity: usize)
        -> Result<Self, TensorError>
    {
        Self::new(*shape.dims(), shape.signed_strides(), origin, capacity)
    }

    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] { &self.dims }
    #[inline(always)]
    pub const fn strides(&self) -> &[isize; R] { &self.strides }
    #[inline(always)]
    pub const fn origin_offset(&self) -> isize { self.origin }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.len }
    /// Logical element count, not the physical accessible span.
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.len }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.len == 0 }
    #[inline(always)]
    pub const fn is_scalar(&self) -> bool { R == 0 }
    #[inline(always)]
    pub const fn capacity(&self) -> usize { self.capacity }
    #[inline(always)]
    pub const fn nbytes(&self) -> usize { self.len * core::mem::size_of::<f32>() }

    #[inline]
    pub const fn reachable_bounds(&self) -> Option<(usize, usize)> {
        match (self.low, self.high) {
            (Some(low), Some(high)) => Some((low, high)),
            _ => None,
        }
    }

    /// The smallest physical bounding span (not the number of logical values).
    #[inline]
    pub fn span_len(&self) -> usize {
        match self.reachable_bounds() {
            Some((low, high)) => high - low + 1,
            None => 0,
        }
    }

    /// Checked logical index -> element offset, valid for all signed strides.
    /// Pointer arithmetic is left to a subsequent, separately borrowed slice.
    #[inline]
    pub fn checked_offset(&self, index: [usize; R]) -> Result<usize, TensorError> {
        if self.len == 0 { return Err(TensorError::IndexOutOfBounds); }
        let mut offset = self.origin as i128;
        for axis in 0..R {
            if index[axis] >= self.dims[axis] {
                return Err(TensorError::IndexOutOfBounds);
            }
            let term = (index[axis] as i128)
                .checked_mul(self.strides[axis] as i128)
                .ok_or(TensorError::ShapeOverflow)?;
            offset = offset.checked_add(term).ok_or(TensorError::ShapeOverflow)?;
        }
        let offset = usize::try_from(offset).map_err(|_| TensorError::IndexOutOfBounds)?;
        if offset >= self.capacity { return Err(TensorError::IndexOutOfBounds); }
        Ok(offset)
    }

    pub fn internal_overlap(&self) -> InternalOverlap {
        // Share the exact tri-state classifier with Shape and mutable views.
        super::signed_mapping_overlap(&self.dims, &self.strides, self.len)
    }

    #[inline]
    pub fn is_unique(&self) -> bool { self.internal_overlap() == InternalOverlap::None }

    #[inline]
    pub fn is_exhaustive(&self) -> bool {
        self.is_unique() && self.len == self.span_len()
    }

    pub fn is_c_contiguous(&self) -> bool { self.is_forward_contiguous(false) }
    pub fn is_f_contiguous(&self) -> bool { self.is_forward_contiguous(true) }

    fn is_forward_contiguous(&self, column_major: bool) -> bool {
        if self.len == 0 { return true; }
        let mut expected = 1usize;
        for pos in 0..R {
            let axis = if column_major { pos } else { R - 1 - pos };
            if self.dims[axis] > 1
                && (self.strides[axis] < 0 || self.strides[axis] as usize != expected)
            {
                return false;
            }
            let Some(next) = expected.checked_mul(self.dims[axis]) else { return false; };
            expected = next;
        }
        true
    }

    /// The only physical slice matching *logical* C-order. Reverse-stride and
    /// broadcast mappings are not writable/readable as a flat Rust slice.
    pub fn contiguous_range(&self) -> Result<core::ops::Range<usize>, TensorError> {
        if !self.is_c_contiguous() { return Err(TensorError::NonContiguous); }
        let start = usize::try_from(self.origin).map_err(|_| TensorError::IndexOutOfBounds)?;
        let end = start.checked_add(self.len).ok_or(TensorError::ShapeOverflow)?;
        if end > self.capacity { return Err(TensorError::IndexOutOfBounds); }
        Ok(start..end)
    }
}

// Tensor remains the primary public type; canonical signed layout metadata
// is an explicit zero-copy query on that type, not a second tensor family.
impl<const R: usize> super::Tensor<f32, R>
where
    Rank<R>: ValidRank,
{
    pub fn try_layout(&self) -> Result<Layout<R>, TensorError> {
        Layout::from_shape(*self.shape(), 0, self.len())
    }
}

impl<'a, const R: usize> super::TensorView<'a, f32, R>
where
    Rank<R>: ValidRank,
{
    pub fn try_layout(&self) -> Result<Layout<R>, TensorError> {
        Layout::from_shape(*self.shape(), 0, self.shape().span_len())
    }
}

impl<'a, const R: usize> super::TensorViewMut<'a, f32, R>
where
    Rank<R>: ValidRank,
{
    pub fn try_layout(&self) -> Result<Layout<R>, TensorError> {
        Layout::from_shape(*self.shape(), 0, self.shape().span_len())
    }
}
