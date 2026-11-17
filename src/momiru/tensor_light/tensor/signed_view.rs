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
// Subcomponent:  Signed View
// Module:        momiru::tensor_light::tensor::signed_view
// Source:        src/tensor_light/tensor/signed_view.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Safe borrowed f32 tensor views with validated signed strides and zero-copy
//   access.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Validate logical addresses within a trusted borrowed host slice.
//   - Avoid forming invalid pointers for empty views.
//   - Provide signed-stride metadata transformations without bulk computation.
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

//! Safe, zero-copy, signed-stride f32 views. No raw pointer is formed for an
//! empty view. Every nonempty logical address is checked against one host slice.
//!
//! This is pure metadata construction; compute stays with the frozen UASDEA plan.
use super::{InternalOverlap, Layout, Rank, Tensor, TensorError, ValidRank};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedSliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

impl SignedSliceSpec {
    pub const fn new(start: Option<isize>, stop: Option<isize>, step: isize) -> Self {
        Self { start, stop, step }
    }
    pub const fn reverse() -> Self { Self::new(None, None, -1) }
    pub const fn full() -> Self { Self::new(None, None, 1) }
}

/// Borrowed read-only host view. Negative strides and zero-stride broadcast
/// are safe because only shared references can escape this representation.
pub struct SignedTensorView<'a, const R: usize>
where Rank<R>: ValidRank,
{
    layout: Layout<R>,
    storage: &'a [f32],
}

impl<'a, const R: usize> SignedTensorView<'a, R>
where Rank<R>: ValidRank,
{
    /// Construct from an already safely borrowed backing slice, signed strides
    /// measured in f32 elements and an origin from the start of that slice.
    pub fn new(
        dims: [usize; R], strides: [isize; R], origin: isize, storage: &'a [f32],
    ) -> Result<Self, TensorError> {
        let layout = Layout::new(dims, strides, origin, storage.len())?;
        Ok(Self { layout, storage })
    }

    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] { self.layout.dims() }
    #[inline(always)]
    pub const fn strides(&self) -> &[isize; R] { self.layout.strides() }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.layout.len() }
    /// Logical cardinality, not storage span or allocation capacity.
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.layout.numel() }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.layout.len() == 0 }
    #[inline(always)]
    pub const fn origin_offset(&self) -> isize { self.layout.origin_offset() }

    #[inline(always)]
    pub const fn is_scalar(&self) -> bool { R == 0 }

    /// Conservative physical overlap classification, independent of the
    /// relative logical traversal order (negative strides are supported).
    #[inline]
    pub fn internal_overlap(&self) -> InternalOverlap { self.layout.internal_overlap() }

    #[inline]
    pub fn is_unique(&self) -> bool { self.layout.is_unique() }

    #[inline]
    pub fn is_exhaustive(&self) -> bool { self.layout.is_exhaustive() }

    #[inline]
    pub fn is_c_contiguous(&self) -> bool { self.layout.is_c_contiguous() }

    #[inline]
    pub fn is_f_contiguous(&self) -> bool { self.layout.is_f_contiguous() }

    #[inline(always)]
    pub const fn layout(&self) -> &Layout<R> { &self.layout }

    /// Guaranteed-safe immutable element reference. Nonempty construction
    /// proved all address bounds, and this check proves the index is in shape.
    #[inline]
    pub fn try_get_ref(&self, index: [usize; R]) -> Result<&'a f32, TensorError> {
        let offset = self.layout.checked_offset(index)?;
        self.storage.get(offset).ok_or(TensorError::IndexOutOfBounds)
    }

    #[inline]
    pub fn get(&self, index: [usize; R]) -> Result<f32, TensorError> {
        Ok(*self.try_get_ref(index)?)
    }

    /// Only a physically forward, C-contiguous logical region exposes a slice.
    pub fn as_slice(&self) -> Result<&'a [f32], TensorError> {
        let range = self.layout.contiguous_range()?;
        self.storage.get(range).ok_or(TensorError::IndexOutOfBounds)
    }

    /// Read-only, right-aligned NumPy-style broadcasting for signed-stride
    /// views. Expanded axes receive zero stride; no data is moved. Zero-size
    /// dimensions follow the rule that 0 and 1 broadcast to zero, not one.
    pub fn broadcast_to<const OUT: usize>(
        &self, dims: [usize; OUT],
    ) -> Result<SignedTensorView<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        if OUT < R { return Err(TensorError::ShapeMismatch); }
        let leading = OUT - R;
        let mut strides = [0isize; OUT];
        for axis in leading..OUT {
            let source = axis - leading;
            if self.layout.dims()[source] == dims[axis] {
                strides[axis] = self.layout.strides()[source];
            } else if self.layout.dims()[source] == 1 {
                strides[axis] = 0;
            } else {
                return Err(TensorError::ShapeMismatch);
            }
        }
        SignedTensorView::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    /// Insert one length-one axis without copying data. Rank nine is rejected
    /// by the sealed Rank witness and explicit OUT/R relation.
    pub fn unsqueeze<const OUT: usize>(
        &self, axis: usize,
    ) -> Result<SignedTensorView<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        if R == 8 || OUT != R + 1 || axis > R {
            return Err(TensorError::AxisOutOfBounds);
        }
        let mut dims = [0usize; OUT];
        let mut strides = [0isize; OUT];
        let mut src = 0usize;
        for dst in 0..OUT {
            if dst == axis {
                dims[dst] = 1;
                strides[dst] = 0;
            } else {
                dims[dst] = self.layout.dims()[src];
                strides[dst] = self.layout.strides()[src];
                src += 1;
            }
        }
        SignedTensorView::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    /// Remove exactly one singleton axis while retaining the same owner.
    pub fn squeeze_axis<const OUT: usize>(
        &self, axis: usize,
    ) -> Result<SignedTensorView<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        if R == 0 || OUT + 1 != R || axis >= R || self.layout.dims()[axis] != 1 {
            return Err(TensorError::ShapeMismatch);
        }
        let mut dims = [0usize; OUT];
        let mut strides = [0isize; OUT];
        let mut dst = 0usize;
        for src in 0..R {
            if src != axis {
                dims[dst] = self.layout.dims()[src];
                strides[dst] = self.layout.strides()[src];
                dst += 1;
            }
        }
        SignedTensorView::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    /// Zero-copy C-order reshape only when logical forward traversal already
    /// matches one contiguous host region; reverse/permuted/gapped views are
    /// never silently materialized or reinterpreted with different semantics.
    pub fn reshape_view<const OUT: usize>(
        &self, dims: [usize; OUT],
    ) -> Result<SignedTensorView<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        if !self.is_c_contiguous() { return Err(TensorError::NonContiguous); }
        let shape = super::Shape::<OUT>::contiguous(dims)?;
        if shape.len() != self.layout.len() { return Err(TensorError::ShapeMismatch); }
        SignedTensorView::new(dims, shape.signed_strides(), self.layout.origin_offset(), self.storage)
    }

    pub fn permute(&self, axes: [usize; R]) -> Result<Self, TensorError> {
        let mut seen = [false; R];
        let mut dims = [0usize; R];
        let mut strides = [0isize; R];
        for dst in 0..R {
            let src = axes[dst];
            if src >= R || seen[src] { return Err(TensorError::InvalidPermutation); }
            seen[src] = true;
            dims[dst] = self.layout.dims()[src];
            strides[dst] = self.layout.strides()[src];
        }
        Self::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, TensorError> {
        if a >= R || b >= R { return Err(TensorError::AxisOutOfBounds); }
        let mut axes = core::array::from_fn(|i| i);
        axes.swap(a,b);
        self.permute(axes)
    }

    pub fn flip(&self, axis: usize) -> Result<Self, TensorError> {
        self.slice_axis(axis, SignedSliceSpec::reverse())
    }

    /// Exchange two axes without copying elements. Signed strides and the
    /// original allocation-relative origin are retained and revalidated.
    #[inline]
    pub fn swapaxes(&self, a: usize, b: usize) -> Result<Self, TensorError> {
        self.transpose(a, b)
    }

    /// Move a source axis to the requested destination while retaining the
    /// relative order of all other axes. This is a pure metadata operation.
    pub fn movedim(&self, from: usize, to: usize) -> Result<Self, TensorError> {
        if from >= R || to >= R { return Err(TensorError::AxisOutOfBounds); }
        let mut axes = core::array::from_fn(|axis| axis);
        let moving = axes[from];
        if from < to {
            for axis in from..to { axes[axis] = axes[axis + 1]; }
        } else {
            for axis in (to + 1..=from).rev() { axes[axis] = axes[axis - 1]; }
        }
        axes[to] = moving;
        self.permute(axes)
    }

    /// Checked nonmaterializing subrange along a single axis. For an empty
    /// result we preserve an abstract origin without constructing a pointer,
    /// including when another dimension is zero and extent/stride are huge.
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        if axis >= R { return Err(TensorError::AxisOutOfBounds); }
        let end = start.checked_add(length).ok_or(TensorError::ShapeOverflow)?;
        if end > self.layout.dims()[axis] { return Err(TensorError::IndexOutOfBounds); }
        let mut dims = *self.layout.dims();
        dims[axis] = length;
        if dims.contains(&0) {
            return Self::new(dims, *self.layout.strides(), self.layout.origin_offset(), self.storage);
        }
        let delta = (start as i128)
            .checked_mul(self.layout.strides()[axis] as i128)
            .ok_or(TensorError::ShapeOverflow)?;
        let origin = (self.layout.origin_offset() as i128)
            .checked_add(delta).ok_or(TensorError::ShapeOverflow)?;
        let origin = isize::try_from(origin).map_err(|_| TensorError::ShapeOverflow)?;
        Self::new(dims, *self.layout.strides(), origin, self.storage)
    }

    /// Merge the selected adjacent axes when (and only when) their logical
    /// C-order index progression admits one exact affine signed stride.
    /// The rest of the view may be transposed, strided, broadcast, or gapped.
    /// Negative contiguous progression is allowed; incompatible mixed steps
    /// are rejected. No copy, GPU launch or runtime strategy selection occurs.
    pub fn flatten_view<const OUT: usize>(
        &self, start: usize, end: usize,
    ) -> Result<SignedTensorView<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        if R == 0 || start > end || end >= R || OUT != R - (end - start) {
            return Err(TensorError::ShapeMismatch);
        }
        let (merged, merged_stride) = super::collapse_flatten_group(
            self.layout.dims(), self.layout.strides(), start, end,
        )?;
        let mut dims = [0usize; OUT];
        let mut strides = [0isize; OUT];
        let mut dst = 0usize;
        for src in 0..R {
            if src == start {
                dims[dst] = merged;
                strides[dst] = merged_stride;
                dst += 1;
            } else if src < start || src > end {
                dims[dst] = self.layout.dims()[src];
                strides[dst] = self.layout.strides()[src];
                dst += 1;
            }
        }
        SignedTensorView::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    /// Python-like half-open slicing, including negative step. `None` stop
    /// for step<0 means the sentinel -1 (not the explicit index -1).
    pub fn slice_axis(&self, axis: usize, spec: SignedSliceSpec) -> Result<Self, TensorError> {
        if axis >= R { return Err(TensorError::AxisOutOfBounds); }
        if spec.step == 0 { return Err(TensorError::InvalidSliceStep); }
        let n = i128::try_from(self.layout.dims()[axis]).map_err(|_| TensorError::ShapeOverflow)?;
        let step = i128::from(spec.step as i64);
        let (start, stop) = if step > 0 {
            (normal_slice_endpoint(spec.start, n, 0, 0, n),
             normal_slice_endpoint(spec.stop, n, n, 0, n))
        } else {
            (normal_slice_endpoint(spec.start, n, n-1, -1, n-1),
             normal_slice_endpoint(spec.stop, n, -1, -1, n-1))
        };
        let count = if step > 0 && start < stop {
            ((stop - start - 1)/step + 1) as usize
        } else if step < 0 && start > stop {
            ((start - stop - 1)/(-step) + 1) as usize
        } else { 0 };
        let mut dims = *self.layout.dims();
        dims[axis] = count;
        let mut strides = *self.layout.strides();
        if count == 0 || self.layout.len() == 0 {
            // No reachable f32 address: keep a valid structural empty origin.
            strides[axis] = 0;
            return Self::new(dims, strides, self.layout.origin_offset(), self.storage);
        }
        let orig = i128::from(self.layout.origin_offset() as i64);
        let delta = start.checked_mul(i128::from(self.layout.strides()[axis] as i64))
            .ok_or(TensorError::ShapeOverflow)?;
        let new_origin = orig.checked_add(delta).ok_or(TensorError::ShapeOverflow)?;
        // A singleton has no within-axis motion. Canonical stride zero avoids
        // spuriously rejecting a valid one-element slice merely because a
        // user-specified very large step would overflow stride * step.
        // For multiple elements the full checked signed stride is necessary.
        if count == 1 {
            strides[axis] = 0;
        } else {
            let new_stride = i128::from(self.layout.strides()[axis] as i64)
                .checked_mul(step).ok_or(TensorError::ShapeOverflow)?;
            strides[axis] = isize::try_from(new_stride).map_err(|_| TensorError::ShapeOverflow)?;
        }
        Self::new(dims, strides,
            isize::try_from(new_origin).map_err(|_| TensorError::ShapeOverflow)?, self.storage)
    }
}

/// The semantics of explicit negative endpoints differ from omitted endpoints.
fn normal_slice_endpoint(value: Option<isize>, n: i128, default: i128, low: i128, high: i128) -> i128 {
    match value {
        None => default,
        Some(raw) => {
            let mut idx = i128::from(raw as i64);
            if idx < 0 { idx += n; }
            idx.clamp(low, high)
        }
    }
}

impl<const R: usize> Tensor<f32, R>
where Rank<R>: ValidRank,
{
    pub fn signed_view(&self) -> Result<SignedTensorView<'_, R>, TensorError> {
        let mut strides = [0isize; R];
        for (out, &value) in strides.iter_mut().zip(self.shape().strides().iter()) {
            *out = isize::try_from(value).map_err(|_| TensorError::ShapeOverflow)?;
        }
        SignedTensorView::new(*self.shape().dims(), strides, 0, self.as_slice())
    }

    pub fn flip_view(&self, axis: usize) -> Result<SignedTensorView<'_, R>, TensorError> {
        self.signed_view()?.flip(axis)
    }

    /// Direct Tensor-first read-only signed slice. This is *metadata only*;
    /// it never copies data, compiles a kernel, or resolves a new execution plan.
    #[inline]
    pub fn slice_axis_signed(&self, axis: usize, spec: SignedSliceSpec)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.slice_axis(axis, spec)
    }

    /// Reverse one axis as a checked, read-only zero-copy view, including
    /// negative-stride layouts. The owned Tensor remains the primary API.
    #[inline]
    pub fn flip_signed(&self, axis: usize)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.flip(axis)
    }

    /// Signed-layout, read-only swap of two axes; no implicit materialization.
    #[inline]
    pub fn swapaxes_signed(&self, a: usize, b: usize)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.swapaxes(a, b)
    }

    #[inline]
    pub fn permute_signed(&self, axes: [usize; R])
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.permute(axes)
    }

    #[inline]
    pub fn transpose_signed(&self, a: usize, b: usize)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.transpose(a, b)
    }

    #[inline]
    pub fn movedim_signed(&self, from: usize, to: usize)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.movedim(from, to)
    }

    #[inline]
    pub fn narrow_signed(&self, axis: usize, start: usize, length: usize)
        -> Result<SignedTensorView<'_, R>, TensorError>
    {
        self.signed_view()?.narrow(axis, start, length)
    }

    #[inline]
    pub fn flatten_signed<const OUT: usize>(&self, start: usize, end: usize)
        -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view()?.flatten_view::<OUT>(start, end)
    }

    #[inline]
    pub fn reshape_signed<const OUT: usize>(&self, dims: [usize; OUT])
        -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view()?.reshape_view::<OUT>(dims)
    }

    #[inline]
    pub fn unsqueeze_signed<const OUT: usize>(&self, axis: usize)
        -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view()?.unsqueeze::<OUT>(axis)
    }

    #[inline]
    pub fn squeeze_axis_signed<const OUT: usize>(&self, axis: usize)
        -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view()?.squeeze_axis::<OUT>(axis)
    }

    /// Broadcast remains read-only when zero strides create overlapping
    /// logical indices. There is no implicit output allocation or copy.
    #[inline]
    pub fn broadcast_signed<const OUT: usize>(&self, dims: [usize; OUT])
        -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view()?.broadcast_to::<OUT>(dims)
    }

    /// Construct an arbitrary signed-stride, read-only zero-copy view directly
    /// from the canonical Tensor API. Bounds and origin are checked against
    /// the owned allocation; zero-stride broadcasts are read-only.
    pub fn as_strided_signed<const OUT: usize>(
        &self, dims: [usize; OUT], strides: [isize; OUT], origin: isize,
    ) -> Result<SignedTensorView<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        SignedTensorView::new(dims, strides, origin, self.as_slice())
    }
}
