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
// Subcomponent:  View
// Module:        momiru::tensor_light::tensor::view
// Source:        src/tensor_light/tensor/view.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Borrowed f32 tensor views and zero-copy nonnegative-stride metadata
//   transformations.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Represent shapes and strides over borrowed tensor storage.
//   - Support transpose, reshape, expand, narrow and positive-step slicing.
//   - Avoid silent data materialization for metadata-only transformations.
//   - Keep signed negative-stride representations consistent with plan layouts.
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

//! Borrowed tensor views and zero-copy non-negative-stride view operations.
//!
//! A view carries shape/stride metadata over borrowed storage. Transpose,
//! reshape, expand/broadcast, narrow and positive-step slices are metadata-only.
//! Reverse (negative-stride) views are represented by the UASDEA engine's
//! signed physical layout/base-offset contract; this host convenience view keeps
//! `usize` strides so safe borrowed slices never point before their base.

use super::{
    normalize_axis, normalize_index, InternalOverlap, Rank, Shape, Tensor, TensorElement, TensorError, ValidRank,
};

pub struct TensorView<'a, T: TensorElement, const R: usize>
where
    Rank<R>: ValidRank,
{
    pub(super) shape: Shape<R>,
    pub(super) data: &'a [T],
}

#[inline]
fn permuted_shape<const R: usize>(
    shape: Shape<R>,
    axes: [usize; R],
) -> Result<Shape<R>, TensorError>
where
    Rank<R>: ValidRank,
{
    let mut seen = [false; R];
    let mut dims = [0usize; R];
    let mut strides = [0usize; R];
    let mut out_axis = 0usize;
    while out_axis < R {
        let source_axis = axes[out_axis];
        if source_axis >= R || seen[source_axis] {
            return Err(TensorError::InvalidPermutation);
        }
        seen[source_axis] = true;
        dims[out_axis] = shape.dims()[source_axis];
        strides[out_axis] = shape.strides()[source_axis];
        out_axis += 1;
    }
    Shape::strided(dims, strides)
}

#[inline]
fn reshape_view_from_parts<'a, T: TensorElement, const R: usize, const R2: usize>(
    shape: Shape<R>,
    data: &'a [T],
    dims: [usize; R2],
) -> Result<TensorView<'a, T, R2>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<R2>: ValidRank,
{
    if !shape.is_contiguous() {
        return Err(TensorError::ShapeMismatch);
    }
    let output = Shape::<R2>::contiguous(dims)?;
    if output.len() != shape.len() {
        return Err(TensorError::ShapeMismatch);
    }
    TensorView::new(output, &data[..shape.len()])
}

#[inline]
fn expand_view_from_parts<'a, T: TensorElement, const R: usize, const R2: usize>(
    shape: Shape<R>,
    data: &'a [T],
    dims: [usize; R2],
) -> Result<TensorView<'a, T, R2>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<R2>: ValidRank,
{
    if R2 < R {
        return Err(TensorError::ShapeMismatch);
    }
    let leading = R2 - R;
    let mut strides = [0usize; R2];
    let mut axis = 0usize;
    while axis < R2 {
        if axis < leading {
            strides[axis] = 0;
        } else {
            let source_axis = axis - leading;
            let source_dim = shape.dims()[source_axis];
            let target_dim = dims[axis];
            if source_dim == target_dim {
                strides[axis] = shape.strides()[source_axis];
            } else if source_dim == 1 {
                strides[axis] = 0;
            } else {
                return Err(TensorError::ShapeMismatch);
            }
        }
        axis += 1;
    }
    TensorView::new(Shape::strided(dims, strides)?, data)
}

/// Metadata-only insertion of one length-one axis. No generic const
/// arithmetic in types: OUT is explicitly supplied and validated at runtime
/// (with both ranks statically constrained to 0..8).
fn unsqueeze_parts<'a, T: TensorElement, const R: usize, const OUT: usize>(
    shape: Shape<R>, data: &'a [T], axis: usize,
) -> Result<TensorView<'a, T, OUT>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<OUT>: ValidRank,
{
    if R >= 8 || OUT != R + 1 || axis > R { return Err(TensorError::AxisOutOfBounds); }
    let mut dims = [0usize; OUT];
    let mut strides = [0usize; OUT];
    let mut src = 0usize;
    for dst in 0..OUT {
        if dst == axis {
            dims[dst] = 1;
            strides[dst] = 0;
        } else {
            dims[dst] = shape.dims()[src];
            strides[dst] = shape.strides()[src];
            src += 1;
        }
    }
    TensorView::new(Shape::strided(dims, strides)?, data)
}

fn squeeze_axis_parts<'a, T: TensorElement, const R: usize, const OUT: usize>(
    shape: Shape<R>, data: &'a [T], axis: usize,
) -> Result<TensorView<'a, T, OUT>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<OUT>: ValidRank,
{
    if R == 0 || OUT + 1 != R || axis >= R || shape.dims()[axis] != 1 {
        return Err(TensorError::ShapeMismatch);
    }
    let mut dims = [0usize; OUT];
    let mut strides = [0usize; OUT];
    let mut dst = 0usize;
    for src in 0..R {
        if src != axis {
            dims[dst] = shape.dims()[src];
            strides[dst] = shape.strides()[src];
            dst += 1;
        }
    }
    TensorView::new(Shape::strided(dims, strides)?, data)
}

fn flatten_parts<'a, T: TensorElement, const R: usize, const OUT: usize>(
    shape: Shape<R>, data: &'a [T], start: usize, end: usize,
) -> Result<TensorView<'a, T, OUT>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<OUT>: ValidRank,
{
    if R == 0 || start > end || end >= R || OUT != R - (end - start) {
        return Err(TensorError::ShapeMismatch);
    }
    // Only the selected axis group must be flattenable. Outer axes may be
    // transposed or have gaps; a frozen dataflow graph is not needed for this
    // exact metadata-only remapping.
    let (collapsed, collapsed_stride) = super::collapse_flatten_group(
        shape.dims(), &shape.signed_strides(), start, end,
    )?;
    let mut dims = [0usize; OUT];
    let mut strides = [0usize; OUT];
    let mut dst = 0usize;
    for src in 0..R {
        if src == start {
            dims[dst] = collapsed;
            // All positive strides entering Shape were checked to fit isize.
            strides[dst] = usize::try_from(collapsed_stride)
                .map_err(|_| TensorError::NonContiguous)?;
            dst += 1;
        } else if src < start || src > end {
            dims[dst] = shape.dims()[src];
            strides[dst] = shape.strides()[src];
            dst += 1;
        }
    }
    TensorView::new(Shape::<OUT>::strided(dims, strides)?, data)
}

fn movedim_axes<const R: usize>(from: usize, to: usize) -> Result<[usize; R], TensorError> {
    if from >= R || to >= R { return Err(TensorError::AxisOutOfBounds); }
    let mut axes = core::array::from_fn(|i| i);
    let moving = axes[from];
    if from < to {
        for i in from..to { axes[i] = axes[i + 1]; }
    } else {
        for i in (to + 1..=from).rev() { axes[i] = axes[i - 1]; }
    }
    axes[to] = moving;
    Ok(axes)
}

impl<'a, T: TensorElement, const R: usize> TensorView<'a, T, R>
where
    Rank<R>: ValidRank,
{
    pub fn new(shape: Shape<R>, data: &'a [T]) -> Result<Self, TensorError> {
        let span = shape.span_len();
        if data.len() < span {
            return Err(TensorError::LengthMismatch);
        }
        Ok(Self {
            shape,
            data: &data[..span],
        })
    }

    pub fn new_strided(
        dims: [usize; R],
        strides: [usize; R],
        data: &'a [T],
    ) -> Result<Self, TensorError> {
        let shape = Shape::strided(dims, strides)?;
        Self::new(shape, data)
    }

    #[inline(always)]
    pub const fn shape(&self) -> &Shape<R> {
        &self.shape
    }

    /// Constant-cost metadata helpers; never expose a strided view as a flat
    /// contiguous slice, and never perform payload reads or bulk operations.
    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] { self.shape.dims() }
    #[inline(always)]
    pub const fn strides(&self) -> &[usize; R] { self.shape.strides() }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.shape.len() }
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.shape.numel() }
    #[inline(always)]
    pub const fn nbytes(&self) -> usize { self.shape.nbytes() }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.shape.is_empty() }
    #[inline(always)]
    pub const fn is_scalar(&self) -> bool { R == 0 }

    /// Number of elements in the smallest reachable backing span. This may
    /// differ from `numel` for gapped/broadcast views; no allocation is made.
    #[inline(always)]
    pub const fn span_len(&self) -> usize { self.shape.span_len() }

    /// Host metadata checks on the same sealed Shape certificate as Tensor.
    /// These never touch f32 payloads or select/rewrite an execution plan.
    #[inline(always)]
    pub fn is_c_contiguous(&self) -> bool { self.shape.is_c_contiguous() }
    #[inline(always)]
    pub fn is_f_contiguous(&self) -> bool { self.shape.is_f_contiguous() }
    #[inline(always)]
    pub fn internal_overlap(&self) -> InternalOverlap { self.shape.internal_overlap() }
    #[inline(always)]
    pub fn is_unique(&self) -> bool { self.shape.is_unique() }
    #[inline(always)]
    pub fn is_exhaustive(&self) -> bool { self.shape.is_exhaustive() }

    /// Read-only logical C-order slice; reject noncontiguous/strided layouts.
    #[inline(always)]
    pub fn as_slice(&self) -> Result<&'a [T], TensorError> {
        self.as_contiguous_slice().ok_or(TensorError::NonContiguous)
    }

    /// Returns the logical elements as a slice only when this view is contiguous.
    #[inline(always)]
    pub fn as_contiguous_slice(&self) -> Option<&'a [T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[..self.shape.len()])
        } else {
            None
        }
    }

    /// Zero-copy axis permutation. No element is moved.
    pub fn permute(&self, axes: [usize; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        TensorView::new(permuted_shape(self.shape, axes)?, self.data)
    }

    /// Alias for `transpose`, preserving zero-copy semantics.
    pub fn swapaxes(&self, a: isize, b: isize) -> Result<TensorView<'_, T, R>, TensorError> {
        self.transpose(a, b)
    }

    pub fn movedim(&self, from: usize, to: usize) -> Result<TensorView<'_, T, R>, TensorError> {
        self.permute(movedim_axes::<R>(from, to)?)
    }

    pub fn unsqueeze<const OUT: usize>(&self, axis: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        unsqueeze_parts(self.shape, self.data, axis)
    }

    pub fn squeeze_axis<const OUT: usize>(&self, axis: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        squeeze_axis_parts(self.shape, self.data, axis)
    }

    pub fn flatten_view<const OUT: usize>(&self, start: usize, end: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        flatten_parts(self.shape, self.data, start, end)
    }

    pub fn broadcast_to<const OUT: usize>(&self, dims: [usize; OUT])
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.expand_view(dims)
    }

    /// Zero-copy swap of two logical axes.
    pub fn transpose(
        &self,
        dim0: isize,
        dim1: isize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        let d0 = normalize_axis(dim0, R)?;
        let d1 = normalize_axis(dim1, R)?;
        let mut axes = core::array::from_fn(|i| i);
        axes.swap(d0, d1);
        self.permute(axes)
    }

    /// Zero-copy reshape when the source view is logically contiguous.
    pub fn reshape_view<const R2: usize>(
        &self,
        dims: [usize; R2],
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        reshape_view_from_parts(self.shape, self.data, dims)
    }

    /// Zero-copy right-aligned broadcast using stride zero on expanded axes.
    pub fn expand_view<const R2: usize>(
        &self,
        dims: [usize; R2],
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        expand_view_from_parts(self.shape, self.data, dims)
    }

    /// Checked zero-copy arbitrary non-negative-stride view relative to this
    /// view's logical origin. The engine-level UASDEA representation also
    /// supports signed strides; this safe borrowed host surface intentionally
    /// keeps `usize` strides so it never creates a pointer before its base.
    pub fn as_strided_view<const R2: usize>(
        &self,
        dims: [usize; R2],
        strides: [usize; R2],
        storage_offset: usize,
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        TensorView::from_offset_strided(dims, strides, storage_offset, self.data)
    }
}

pub struct TensorViewMut<'a, T: TensorElement, const R: usize>
where
    Rank<R>: ValidRank,
{
    pub(super) shape: Shape<R>,
    pub(super) data: &'a mut [T],
}

impl<'a, T: TensorElement, const R: usize> TensorViewMut<'a, T, R>
where
    Rank<R>: ValidRank,
{
    pub fn new(shape: Shape<R>, data: &'a mut [T]) -> Result<Self, TensorError> {
        if !shape.is_non_overlapping_for_mutation() {
            return Err(TensorError::OverlappingMutableView);
        }
        let span = shape.span_len();
        if data.len() < span {
            return Err(TensorError::LengthMismatch);
        }
        let (view_span, _) = data.split_at_mut(span);
        Ok(Self {
            shape,
            data: view_span,
        })
    }

    pub fn new_strided(
        dims: [usize; R],
        strides: [usize; R],
        data: &'a mut [T],
    ) -> Result<Self, TensorError> {
        let shape = Shape::strided(dims, strides)?;
        Self::new(shape, data)
    }

    /// Read-only arbitrary strided subview. A mutable counterpart is
    /// intentionally not exposed because arbitrary strides may overlap.
    pub fn as_strided_view<const R2: usize>(
        &self,
        dims: [usize; R2],
        strides: [usize; R2],
        storage_offset: usize,
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        TensorView::from_offset_strided(dims, strides, storage_offset, self.data)
    }

    #[inline(always)]
    pub const fn shape(&self) -> &Shape<R> {
        &self.shape
    }

    /// Constant-cost metadata helpers; never expose a strided view as a flat
    /// contiguous slice, and never perform payload reads or bulk operations.
    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] { self.shape.dims() }
    #[inline(always)]
    pub const fn strides(&self) -> &[usize; R] { self.shape.strides() }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.shape.len() }
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.shape.numel() }
    #[inline(always)]
    pub const fn nbytes(&self) -> usize { self.shape.nbytes() }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.shape.is_empty() }
    #[inline(always)]
    pub const fn is_scalar(&self) -> bool { R == 0 }

    /// Number of elements in the smallest reachable backing span. This may
    /// differ from `numel` for gapped/broadcast views; no allocation is made.
    #[inline(always)]
    pub const fn span_len(&self) -> usize { self.shape.span_len() }

    /// Host metadata checks on the same sealed Shape certificate as Tensor.
    /// These never touch f32 payloads or select/rewrite an execution plan.
    #[inline(always)]
    pub fn is_c_contiguous(&self) -> bool { self.shape.is_c_contiguous() }
    #[inline(always)]
    pub fn is_f_contiguous(&self) -> bool { self.shape.is_f_contiguous() }
    #[inline(always)]
    pub fn internal_overlap(&self) -> InternalOverlap { self.shape.internal_overlap() }
    #[inline(always)]
    pub fn is_unique(&self) -> bool { self.shape.is_unique() }
    #[inline(always)]
    pub fn is_exhaustive(&self) -> bool { self.shape.is_exhaustive() }

    /// Return logical C-order storage only; do not expose stride gaps.
    #[inline(always)]
    pub fn as_slice(&self) -> Result<&[T], TensorError> {
        self.as_contiguous_slice().ok_or(TensorError::NonContiguous)
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], TensorError> {
        self.as_contiguous_mut_slice().ok_or(TensorError::NonContiguous)
    }

    #[inline(always)]
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[..self.shape.len()])
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn as_contiguous_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.shape.is_contiguous() {
            let len = self.shape.len();
            Some(&mut self.data[..len])
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: usize,
}

impl SliceSpec {
    pub const FULL: Self = Self {
        start: None,
        end: None,
        step: 1,
    };

    #[inline(always)]
    pub const fn new(start: Option<isize>, end: Option<isize>, step: usize) -> Self {
        Self { start, end, step }
    }

    #[inline(always)]
    pub const fn full() -> Self {
        Self::FULL
    }
}

impl Default for SliceSpec {
    #[inline(always)]
    fn default() -> Self {
        Self::FULL
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NormalizedSlice {
    start: usize,
    len: usize,
    step: usize,
}

#[inline(always)]
fn normalize_bound(
    value: Option<isize>,
    default: usize,
    dim: usize,
) -> Result<usize, TensorError> {
    let Some(raw) = value else {
        return Ok(default);
    };

    if raw >= 0 {
        return Ok((raw as usize).min(dim));
    }
    Ok(dim.saturating_sub(raw.unsigned_abs()))
}

#[inline(always)]
fn normalize_slice(spec: SliceSpec, dim: usize) -> Result<NormalizedSlice, TensorError> {
    if spec.step == 0 {
        return Err(TensorError::InvalidSliceStep);
    }

    let start = normalize_bound(spec.start, 0, dim)?;
    let end = normalize_bound(spec.end, dim, dim)?;
    let len = if end <= start {
        0
    } else {
        1 + (end - start - 1) / spec.step
    };

    Ok(NormalizedSlice {
        start,
        len,
        step: spec.step,
    })
}

impl<const R: usize> Shape<R>
where
    Rank<R>: ValidRank,
{
    fn sliced(&self, specs: [SliceSpec; R]) -> Result<(Self, usize), TensorError> {
        let mut dims = [0usize; R];
        let mut strides = [0usize; R];
        let mut starts = [0usize; R];
        for axis in 0..R {
            let normalized = normalize_slice(specs[axis], self.dims()[axis])?;
            dims[axis] = normalized.len;
            starts[axis] = normalized.start;
            // No reachable element means zero stride metadata is valid,
            // independently of the requested step's magnitude.
            if normalized.len <= 1 || self.is_empty() {
                // A singleton slice never traverses this axis. Its stride
                // is semantically irrelevant, so an enormous step must not
                // overflow the metadata calculation.
                strides[axis] = 0;
            } else {
                strides[axis] = self.strides()[axis]
                    .checked_mul(normalized.step)
                    .ok_or(TensorError::ShapeOverflow)?;
            }
        }
        if dims.contains(&0) {
            // Crucial: do not calculate base pointer arithmetic for an
            // empty logical view. Avoid fake overflow from huge other axes.
            return Ok((Self::strided(dims, [0usize; R])?, 0));
        }
        let mut base_offset = 0usize;
        for axis in 0..R {
            let axis_offset = starts[axis]
                .checked_mul(self.strides()[axis])
                .ok_or(TensorError::ShapeOverflow)?;
            base_offset = base_offset
                .checked_add(axis_offset)
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok((Self::strided(dims, strides)?, base_offset))
    }

    #[inline]
    fn narrowed(
        &self,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<(Self, usize), TensorError> {
        let end = start.checked_add(len).ok_or(TensorError::ShapeOverflow)?;
        if axis >= R || end > self.dims()[axis] {
            return Err(TensorError::IndexOutOfBounds);
        }

        let mut dims = *self.dims();
        dims[axis] = len;
        let offset = if dims.contains(&0) {
            // Even a nonempty narrowed axis has no physical address when
            // another axis is empty; never compute an unrealizable origin.
            0
        } else {
            start
                .checked_mul(self.strides()[axis])
                .ok_or(TensorError::ShapeOverflow)?
        };
        Ok((Self::strided(dims, *self.strides())?, offset))
    }
}

impl<'a, T: TensorElement, const R: usize> TensorView<'a, T, R>
where
    Rank<R>: ValidRank,
{
    /// Builds a checked zero-copy strided view from an element offset.
    pub fn from_offset_strided(
        dims: [usize; R],
        strides: [usize; R],
        offset: usize,
        data: &'a [T],
    ) -> Result<Self, TensorError> {
        let shape = Shape::strided(dims, strides)?;
        if offset > data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let remaining = data.len() - offset;
        if remaining < shape.span_len() {
            return Err(TensorError::LengthMismatch);
        }
        Self::new(shape, &data[offset..])
    }

    pub fn slice(&self, specs: [SliceSpec; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        let (shape, offset) = self.shape.sliced(specs)?;
        if offset > self.data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        TensorView::new(shape, &self.data[offset..])
    }

    /// Rank-preserving slice along one axis.
    pub fn narrow(
        &self,
        axis: isize,
        start: isize,
        len: usize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        let axis = normalize_axis(axis, R)?;
        let start = normalize_narrow_start(start, self.shape.dims()[axis])?;
        let (shape, offset) = self.shape.narrowed(axis, start, len)?;
        TensorView::new(shape, &self.data[offset..])
    }
}

impl<'a, T: TensorElement, const R: usize> TensorViewMut<'a, T, R>
where
    Rank<R>: ValidRank,
{
    pub fn from_offset_strided(
        dims: [usize; R],
        strides: [usize; R],
        offset: usize,
        data: &'a mut [T],
    ) -> Result<Self, TensorError> {
        let shape = Shape::strided(dims, strides)?;
        if offset > data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let remaining = data.len() - offset;
        if remaining < shape.span_len() {
            return Err(TensorError::LengthMismatch);
        }
        let (_, tail) = data.split_at_mut(offset);
        Self::new(shape, tail)
    }

    pub fn slice(&self, specs: [SliceSpec; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        let (shape, offset) = self.shape.sliced(specs)?;
        if offset > self.data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        TensorView::new(shape, &self.data[offset..])
    }

    pub fn slice_mut(
        &mut self,
        specs: [SliceSpec; R],
    ) -> Result<TensorViewMut<'_, T, R>, TensorError> {
        let (shape, offset) = self.shape.sliced(specs)?;
        if offset > self.data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let (_, tail) = self.data.split_at_mut(offset);
        TensorViewMut::new(shape, tail)
    }

    pub fn narrow(
        &self,
        axis: isize,
        start: isize,
        len: usize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        let axis = normalize_axis(axis, R)?;
        let start = normalize_narrow_start(start, self.shape.dims()[axis])?;
        let (shape, offset) = self.shape.narrowed(axis, start, len)?;
        TensorView::new(shape, &self.data[offset..])
    }

    pub fn narrow_mut(
        &mut self,
        axis: isize,
        start: isize,
        len: usize,
    ) -> Result<TensorViewMut<'_, T, R>, TensorError> {
        let axis = normalize_axis(axis, R)?;
        let start = normalize_narrow_start(start, self.shape.dims()[axis])?;
        let (shape, offset) = self.shape.narrowed(axis, start, len)?;
        let (_, tail) = self.data.split_at_mut(offset);
        TensorViewMut::new(shape, tail)
    }

}

impl<T: TensorElement, const R: usize> Tensor<T, R>
where
    Rank<R>: ValidRank,
{
    pub fn permute(&self, axes: [usize; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        TensorView::new(permuted_shape(self.shape, axes)?, self.data.as_slice())
    }

    pub fn swapaxes(&self, a: isize, b: isize) -> Result<TensorView<'_, T, R>, TensorError> {
        self.transpose(a, b)
    }

    pub fn movedim(&self, from: usize, to: usize) -> Result<TensorView<'_, T, R>, TensorError> {
        self.permute(movedim_axes::<R>(from, to)?)
    }

    pub fn unsqueeze<const OUT: usize>(&self, axis: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        unsqueeze_parts(self.shape, self.data.as_slice(), axis)
    }

    pub fn squeeze_axis<const OUT: usize>(&self, axis: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        squeeze_axis_parts(self.shape, self.data.as_slice(), axis)
    }

    pub fn flatten_view<const OUT: usize>(&self, start: usize, end: usize)
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        flatten_parts(self.shape, self.data.as_slice(), start, end)
    }

    pub fn broadcast_to<const OUT: usize>(&self, dims: [usize; OUT])
        -> Result<TensorView<'_, T, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.expand_view(dims)
    }

    pub fn transpose(
        &self,
        dim0: isize,
        dim1: isize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        let d0 = normalize_axis(dim0, R)?;
        let d1 = normalize_axis(dim1, R)?;
        let mut axes = core::array::from_fn(|i| i);
        axes.swap(d0, d1);
        self.permute(axes)
    }

    pub fn reshape_view<const R2: usize>(
        &self,
        dims: [usize; R2],
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        reshape_view_from_parts(self.shape, self.data.as_slice(), dims)
    }

    pub fn expand_view<const R2: usize>(
        &self,
        dims: [usize; R2],
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        expand_view_from_parts(self.shape, self.data.as_slice(), dims)
    }

    /// Checked zero-copy arbitrary non-negative-stride view over this tensor's
    /// backing storage.
    pub fn as_strided_view<const R2: usize>(
        &self,
        dims: [usize; R2],
        strides: [usize; R2],
        storage_offset: usize,
    ) -> Result<TensorView<'_, T, R2>, TensorError>
    where
        Rank<R2>: ValidRank,
    {
        TensorView::from_offset_strided(
            dims,
            strides,
            storage_offset,
            self.data.as_slice(),
        )
    }

    pub fn slice(&self, specs: [SliceSpec; R]) -> Result<TensorView<'_, T, R>, TensorError> {
        let (shape, offset) = self.shape.sliced(specs)?;
        let data = self.data.as_slice();
        if offset > data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        TensorView::new(shape, &data[offset..])
    }

    pub fn slice_mut(
        &mut self,
        specs: [SliceSpec; R],
    ) -> Result<TensorViewMut<'_, T, R>, TensorError> {
        let (shape, offset) = self.shape.sliced(specs)?;
        let data = self.data.as_mut_slice();
        if offset > data.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let (_, tail) = data.split_at_mut(offset);
        TensorViewMut::new(shape, tail)
    }

    pub fn narrow(
        &self,
        axis: isize,
        start: isize,
        len: usize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        let axis = normalize_axis(axis, R)?;
        let start = normalize_narrow_start(start, self.shape.dims()[axis])?;
        let (shape, offset) = self.shape.narrowed(axis, start, len)?;
        let data = self.data.as_slice();
        TensorView::new(shape, &data[offset..])
    }

    pub fn narrow_mut(
        &mut self,
        axis: isize,
        start: isize,
        len: usize,
    ) -> Result<TensorViewMut<'_, T, R>, TensorError> {
        let axis = normalize_axis(axis, R)?;
        let start = normalize_narrow_start(start, self.shape.dims()[axis])?;
        let (shape, offset) = self.shape.narrowed(axis, start, len)?;
        let data = self.data.as_mut_slice();
        let (_, tail) = data.split_at_mut(offset);
        TensorViewMut::new(shape, tail)
    }

    #[inline]
    pub fn slice_dim0(
        &self,
        start: isize,
        len: usize,
    ) -> Result<TensorView<'_, T, R>, TensorError> {
        self.narrow(0, start, len)
    }

}

impl<T: TensorElement> Tensor<T, 1> {
    pub fn head(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let len = n.min(self.shape.dims()[0]);
        self.narrow(0, 0, len)
    }

    pub fn head_mut(&mut self, n: usize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        let len = n.min(self.shape.dims()[0]);
        self.narrow_mut(0, 0, len)
    }

    pub fn tail(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let dim = self.shape.dims()[0];
        let len = n.min(dim);
        let start = isize::try_from(dim - len).map_err(|_| TensorError::ShapeOverflow)?;
        self.narrow(0, start, len)
    }

    pub fn tail_mut(&mut self, n: usize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        let dim = self.shape.dims()[0];
        let len = n.min(dim);
        let start = isize::try_from(dim - len).map_err(|_| TensorError::ShapeOverflow)?;
        self.narrow_mut(0, start, len)
    }
}

impl<T: TensorElement> Tensor<T, 2> {
    #[inline]
    pub fn row(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 0)
    }

    #[inline]
    pub fn row_mut(&mut self, index: isize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        self.plane_mut(index, 0)
    }

    #[inline]
    pub fn col(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 1)
    }

    #[inline]
    pub fn col_mut(&mut self, index: isize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        self.plane_mut(index, 1)
    }
}

impl<'a, T: TensorElement> TensorView<'a, T, 1> {
    pub fn head(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let len = n.min(self.shape.dims()[0]);
        self.narrow(0, 0, len)
    }

    pub fn tail(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let dim = self.shape.dims()[0];
        let len = n.min(dim);
        let start = isize::try_from(dim - len).map_err(|_| TensorError::ShapeOverflow)?;
        self.narrow(0, start, len)
    }
}

impl<'a, T: TensorElement> TensorView<'a, T, 2> {
    #[inline]
    pub fn row(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 0)
    }

    #[inline]
    pub fn col(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 1)
    }
}

impl<'a, T: TensorElement> TensorViewMut<'a, T, 1> {
    pub fn head(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let len = n.min(self.shape.dims()[0]);
        self.narrow(0, 0, len)
    }

    pub fn head_mut(&mut self, n: usize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        let len = n.min(self.shape.dims()[0]);
        self.narrow_mut(0, 0, len)
    }

    pub fn tail(&self, n: usize) -> Result<TensorView<'_, T, 1>, TensorError> {
        let dim = self.shape.dims()[0];
        let len = n.min(dim);
        let start = isize::try_from(dim - len).map_err(|_| TensorError::ShapeOverflow)?;
        self.narrow(0, start, len)
    }

    pub fn tail_mut(&mut self, n: usize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        let dim = self.shape.dims()[0];
        let len = n.min(dim);
        let start = isize::try_from(dim - len).map_err(|_| TensorError::ShapeOverflow)?;
        self.narrow_mut(0, start, len)
    }
}

impl<'a, T: TensorElement> TensorViewMut<'a, T, 2> {
    #[inline]
    pub fn row(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 0)
    }

    #[inline]
    pub fn row_mut(&mut self, index: isize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        self.plane_mut(index, 0)
    }

    #[inline]
    pub fn col(&self, index: isize) -> Result<TensorView<'_, T, 1>, TensorError> {
        self.plane(index, 1)
    }

    #[inline]
    pub fn col_mut(&mut self, index: isize) -> Result<TensorViewMut<'_, T, 1>, TensorError> {
        self.plane_mut(index, 1)
    }
}

macro_rules! impl_plane {
    ($rank:literal, $out_rank:literal) => {
        impl<T: TensorElement> Tensor<T, $rank> {
            pub fn plane(
                &self,
                index: isize,
                axis: isize,
            ) -> Result<TensorView<'_, T, $out_rank>, TensorError> {
                plane_from_parts::<T, $rank, $out_rank>(
                    self.shape,
                    self.data.as_slice(),
                    index,
                    axis,
                )
            }

            pub fn plane_mut(
                &mut self,
                index: isize,
                axis: isize,
            ) -> Result<TensorViewMut<'_, T, $out_rank>, TensorError> {
                plane_from_parts_mut::<T, $rank, $out_rank>(
                    self.shape,
                    self.data.as_mut_slice(),
                    index,
                    axis,
                )
            }
        }

        impl<'a, T: TensorElement> TensorView<'a, T, $rank> {
            pub fn plane(
                &self,
                index: isize,
                axis: isize,
            ) -> Result<TensorView<'_, T, $out_rank>, TensorError> {
                plane_from_parts::<T, $rank, $out_rank>(self.shape, &*self.data, index, axis)
            }
        }

        impl<'a, T: TensorElement> TensorViewMut<'a, T, $rank> {
            pub fn plane(
                &self,
                index: isize,
                axis: isize,
            ) -> Result<TensorView<'_, T, $out_rank>, TensorError> {
                plane_from_parts::<T, $rank, $out_rank>(self.shape, &*self.data, index, axis)
            }

            pub fn plane_mut(
                &mut self,
                index: isize,
                axis: isize,
            ) -> Result<TensorViewMut<'_, T, $out_rank>, TensorError> {
                plane_from_parts_mut::<T, $rank, $out_rank>(
                    self.shape,
                    &mut *self.data,
                    index,
                    axis,
                )
            }
        }
    };
}

impl_plane!(2, 1);
impl_plane!(3, 2);
impl_plane!(4, 3);
impl_plane!(5, 4);
impl_plane!(6, 5);
impl_plane!(7, 6);
impl_plane!(8, 7);

fn plane_from_parts<'a, T: TensorElement, const R: usize, const O: usize>(
    shape: Shape<R>,
    data: &'a [T],
    index: isize,
    axis: isize,
) -> Result<TensorView<'a, T, O>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<O>: ValidRank,
{
    let axis = normalize_axis(axis, R)?;
    let index = normalize_index(index, shape.dims()[axis])?;
    // Determine whether any output element exists BEFORE origin arithmetic.
    // Selecting a nominal plane of an empty source can still be a valid
    // empty view, even for enormous signed/unsigned stride descriptors.
    let mut dims = [0usize; O];
    let mut strides = [0usize; O];
    let mut source_axis = 0usize;
    let mut target_axis = 0usize;
    while source_axis < R {
        if source_axis != axis {
            if target_axis >= O {
                return Err(TensorError::ShapeMismatch);
            }
            dims[target_axis] = shape.dims()[source_axis];
            strides[target_axis] = shape.strides()[source_axis];
            target_axis += 1;
        }
        source_axis += 1;
    }
    if target_axis != O {
        return Err(TensorError::ShapeMismatch);
    }
    if dims.contains(&0) {
        return TensorView::from_offset_strided(dims, [0usize; O], 0, data);
    }
    let offset = index.checked_mul(shape.strides()[axis])
        .ok_or(TensorError::ShapeOverflow)?;
    TensorView::from_offset_strided(dims, strides, offset, data)
}

fn plane_from_parts_mut<'a, T: TensorElement, const R: usize, const O: usize>(
    shape: Shape<R>,
    data: &'a mut [T],
    index: isize,
    axis: isize,
) -> Result<TensorViewMut<'a, T, O>, TensorError>
where
    Rank<R>: ValidRank,
    Rank<O>: ValidRank,
{
    let axis = normalize_axis(axis, R)?;
    let index = normalize_index(index, shape.dims()[axis])?;
    // Determine whether any output element exists BEFORE origin arithmetic.
    // Selecting a nominal plane of an empty source can still be a valid
    // empty view, even for enormous signed/unsigned stride descriptors.
    let mut dims = [0usize; O];
    let mut strides = [0usize; O];
    let mut source_axis = 0usize;
    let mut target_axis = 0usize;
    while source_axis < R {
        if source_axis != axis {
            if target_axis >= O {
                return Err(TensorError::ShapeMismatch);
            }
            dims[target_axis] = shape.dims()[source_axis];
            strides[target_axis] = shape.strides()[source_axis];
            target_axis += 1;
        }
        source_axis += 1;
    }
    if target_axis != O {
        return Err(TensorError::ShapeMismatch);
    }
    if dims.contains(&0) {
        return TensorViewMut::from_offset_strided(dims, [0usize; O], 0, data);
    }
    let offset = index.checked_mul(shape.strides()[axis])
        .ok_or(TensorError::ShapeOverflow)?;
    TensorViewMut::from_offset_strided(dims, strides, offset, data)
}

#[inline(always)]
fn normalize_narrow_start(start: isize, dim: usize) -> Result<usize, TensorError> {
    if start >= 0 {
        let normalized = start as usize;
        if normalized <= dim {
            return Ok(normalized);
        }
        return Err(TensorError::IndexOutOfBounds);
    }
    normalize_index(start, dim)
}
