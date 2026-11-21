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
// Subcomponent:  Signed View Mut
// Module:        momiru::tensor_light::tensor::signed_view_mut
// Source:        src/tensor_light/tensor/signed_view_mut.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Exclusive mutable signed-stride f32 host views with conservative nonaliasing
//   proofs.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Prove or reject mutable-view injectivity and disjoint access.
//   - Preserve exclusive borrowing and validated backing allocation bounds.
//   - Keep bulk numeric operations behind frozen execution plans.
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

//! Safe, exclusive, signed-stride f32 views, with conservative injectivity
//! proof. CPU host accesses only; bulk operations require a frozen plan.
use super::{
    Layout, Rank, SignedSliceSpec, SignedTensorView, Tensor, TensorError, TensorViewMut, ValidRank,
};

/// An exclusive zero-copy f32 view, accepting negative strides only when the
/// *whole* logical index mapping is provably injective. A mutable element
/// reference is tied to `&mut self` and cannot outlive its exclusive borrow.
pub struct SignedTensorViewMut<'a, const R: usize>
where
    Rank<R>: ValidRank,
{
    layout: Layout<R>,
    storage: &'a mut [f32],
}

impl<'a, const R: usize> SignedTensorViewMut<'a, R>
where
    Rank<R>: ValidRank,
{
    pub fn new(
        dims: [usize; R], strides: [isize; R], origin: isize, storage: &'a mut [f32],
    ) -> Result<Self, TensorError> {
        let layout = Layout::new(dims, strides, origin, storage.len())?;
        if !layout.is_unique() {
            return Err(TensorError::OverlappingMutableView);
        }
        Ok(Self { layout, storage })
    }

    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] { self.layout.dims() }
    #[inline(always)]
    pub const fn strides(&self) -> &[isize; R] { self.layout.strides() }
    #[inline(always)]
    pub const fn origin_offset(&self) -> isize { self.layout.origin_offset() }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.layout.len() }
    /// Logical cardinality, not storage span or allocation capacity.
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.layout.numel() }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.layout.len() == 0 }

    #[inline(always)]
    pub const fn layout(&self) -> &Layout<R> { &self.layout }

    #[inline]
    fn checked_offset(&self, index: [usize; R]) -> Result<usize, TensorError> {
        self.layout.checked_offset(index)
    }

    #[inline]
    pub fn get(&self, index: [usize; R]) -> Result<f32, TensorError> {
        Ok(*self.try_get_ref(index)?)
    }

    #[inline]
    pub fn try_get_ref(&self, index: [usize; R]) -> Result<&f32, TensorError> {
        let offset = self.checked_offset(index)?;
        self.storage.get(offset).ok_or(TensorError::IndexOutOfBounds)
    }

    #[inline]
    pub fn try_get_mut(&mut self, index: [usize; R]) -> Result<&mut f32, TensorError> {
        let offset = self.checked_offset(index)?;
        self.storage.get_mut(offset).ok_or(TensorError::IndexOutOfBounds)
    }

    #[inline]
    pub fn set(&mut self, index: [usize; R], value: f32) -> Result<(), TensorError> {
        *self.try_get_mut(index)? = value;
        Ok(())
    }

    pub fn as_slice(&self) -> Result<&[f32], TensorError> {
        let range = self.layout.contiguous_range()?;
        self.storage.get(range).ok_or(TensorError::IndexOutOfBounds)
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [f32], TensorError> {
        let range = self.layout.contiguous_range()?;
        self.storage.get_mut(range).ok_or(TensorError::IndexOutOfBounds)
    }

    /// Immutable view borrowing this exclusive view for only the duration
    /// of the receiver borrow; it can never escape as a mutable alias.
    pub fn as_readonly(&self) -> Result<SignedTensorView<'_, R>, TensorError> {
        // Borrowed shared storage; bounds are still checked by the constructor.
        SignedTensorView::new(*self.layout.dims(), *self.layout.strides(), self.layout.origin_offset(), &*self.storage)
    }

    /// Consumes the exclusive view and returns exactly one transformed owner.
    /// The API never clones a `&mut [f32]` into two mutable view handles.
    pub fn permute_mut(self, axes: [usize; R]) -> Result<Self, TensorError> {
        let mut seen = [false; R];
        let mut dims = [0usize; R];
        let mut strides = [0isize; R];
        for axis in 0..R {
            let src = axes[axis];
            if src >= R || seen[src] { return Err(TensorError::InvalidPermutation); }
            seen[src] = true;
            dims[axis] = self.layout.dims()[src];
            strides[axis] = self.layout.strides()[src];
        }
        Self::new(dims, strides, self.layout.origin_offset(), self.storage)
    }

    /// Reorder two axes of the exclusive view without copying or duplicating
    /// the unique mutable backing borrow. The source view is consumed.
    pub fn transpose_mut(self, a: usize, b: usize) -> Result<Self, TensorError> {
        if a >= R || b >= R { return Err(TensorError::AxisOutOfBounds); }
        let mut axes = core::array::from_fn(|i| i);
        axes.swap(a, b);
        self.permute_mut(axes)
    }

    /// An explicit alias for the consuming, checked transpose operation.
    #[inline]
    pub fn swapaxes_mut(self, a: usize, b: usize) -> Result<Self, TensorError> {
        self.transpose_mut(a, b)
    }

    /// Move one axis while keeping the relative order of all other axes.
    /// No new storage owner, writable alias, or execution plan is created.
    pub fn movedim_mut(self, from: usize, to: usize) -> Result<Self, TensorError> {
        if from >= R || to >= R { return Err(TensorError::AxisOutOfBounds); }
        let mut axes = core::array::from_fn(|axis| axis);
        let moving = axes[from];
        if from < to {
            for axis in from..to { axes[axis] = axes[axis + 1]; }
        } else {
            for axis in (to + 1..=from).rev() { axes[axis] = axes[axis - 1]; }
        }
        axes[to] = moving;
        self.permute_mut(axes)
    }

    /// Checked, zero-copy axis subrange. The read-only normalization is used
    /// only to calculate metadata while the mutable borrow remains exclusive;
    /// the resulting mutable view must pass the SAME injectivity guard.
    pub fn narrow_mut(self, axis: usize, start: usize, length: usize)
        -> Result<Self, TensorError>
    {
        let (dims, strides, origin) = {
            let readonly = SignedTensorView::new(
                *self.layout.dims(), *self.layout.strides(),
                self.layout.origin_offset(), &*self.storage,
            )?;
            let narrowed = readonly.narrow(axis, start, length)?;
            (*narrowed.dims(), *narrowed.strides(), narrowed.origin_offset())
        };
        Self::new(dims, strides, origin, self.storage)
    }

    /// Add one singleton dimension. OUT is checked against the sealed
    /// supported-rank witness, so rank 9 is never exposed.
    pub fn unsqueeze_mut<const OUT: usize>(self, axis: usize)
        -> Result<SignedTensorViewMut<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        let (dims, strides, origin) = {
            let readonly = SignedTensorView::new(
                *self.layout.dims(), *self.layout.strides(),
                self.layout.origin_offset(), &*self.storage,
            )?;
            let expanded = readonly.unsqueeze::<OUT>(axis)?;
            (*expanded.dims(), *expanded.strides(), expanded.origin_offset())
        };
        SignedTensorViewMut::new(dims, strides, origin, self.storage)
    }

    /// Remove one singleton dimension without materializing data.
    pub fn squeeze_axis_mut<const OUT: usize>(self, axis: usize)
        -> Result<SignedTensorViewMut<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        let (dims, strides, origin) = {
            let readonly = SignedTensorView::new(
                *self.layout.dims(), *self.layout.strides(),
                self.layout.origin_offset(), &*self.storage,
            )?;
            let reduced = readonly.squeeze_axis::<OUT>(axis)?;
            (*reduced.dims(), *reduced.strides(), reduced.origin_offset())
        };
        SignedTensorViewMut::new(dims, strides, origin, self.storage)
    }

    /// Zero-copy C-order merging of neighboring axes. Nonempty views require
    /// an affine progression through the group. Empty views only require a
    /// representable merged extent.
    pub fn flatten_view_mut<const OUT: usize>(self, start: usize, end: usize)
        -> Result<SignedTensorViewMut<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        let (dims, strides, origin) = {
            let readonly = SignedTensorView::new(
                *self.layout.dims(), *self.layout.strides(),
                self.layout.origin_offset(), &*self.storage,
            )?;
            let flattened = readonly.flatten_view::<OUT>(start, end)?;
            (*flattened.dims(), *flattened.strides(), flattened.origin_offset())
        };
        SignedTensorViewMut::new(dims, strides, origin, self.storage)
    }

    /// A zero-copy reshape is allowed only when the source has logical
    /// forward C-contiguous order and equal element count. The output's
    /// injectivity and capacity are rechecked rather than assumed.
    pub fn reshape_view_mut<const OUT: usize>(self, dims: [usize; OUT])
        -> Result<SignedTensorViewMut<'a, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        let (strides, origin) = {
            let readonly = SignedTensorView::new(
                *self.layout.dims(), *self.layout.strides(),
                self.layout.origin_offset(), &*self.storage,
            )?;
            let reshaped = readonly.reshape_view::<OUT>(dims)?;
            (*reshaped.strides(), reshaped.origin_offset())
        };
        SignedTensorViewMut::new(dims, strides, origin, self.storage)
    }

    /// Consuming slice transform also admits a negative `step` without
    /// creating aliasing mutable handles. The parent view is moved away.
    pub fn slice_axis_mut(self, axis: usize, spec: SignedSliceSpec) -> Result<Self, TensorError> {
        let (dims, strides, origin) = {
            let readonly = SignedTensorView::new(*self.layout.dims(), *self.layout.strides(), self.layout.origin_offset(), &*self.storage)?;
            let sliced = readonly.slice_axis(axis, spec)?;
            (*sliced.dims(), *sliced.strides(), sliced.origin_offset())
        };
        Self::new(dims, strides, origin, self.storage)
    }

    pub fn flip_mut(self, axis: usize) -> Result<Self, TensorError> {
        self.slice_axis_mut(axis, SignedSliceSpec::reverse())
    }
}

impl<'a, const R: usize> TensorViewMut<'a, f32, R>
where
    Rank<R>: ValidRank,
{
    /// Reborrow this view's backing span with signed-stride metadata. Offsets
    /// are relative to the start of this view's span. The result holds the
    /// exclusive borrow for as long as it is in use.
    ///
    /// ```compile_fail
    /// use momiru::tensor_light::TensorViewMut;
    ///
    /// let mut storage = [1.0_f32, 2.0];
    /// let mut view = TensorViewMut::new_strided([2], [1], &mut storage).unwrap();
    /// let mut signed = view.signed_view_mut().unwrap();
    /// let _ = view.signed_view_mut().unwrap();
    /// signed.set([0], 3.0).unwrap();
    /// ```
    pub fn signed_view_mut(&mut self) -> Result<SignedTensorViewMut<'_, R>, TensorError> {
        SignedTensorViewMut::new(*self.shape.dims(), self.shape.signed_strides(), 0, self.data)
    }
}

impl<const R: usize> Tensor<f32, R>
where
    Rank<R>: ValidRank,
{
    pub fn signed_view_mut(&mut self) -> Result<SignedTensorViewMut<'_, R>, TensorError> {
        let dims = *self.shape().dims();
        let strides = self.shape().signed_strides();
        SignedTensorViewMut::new(dims, strides, 0, self.as_mut_slice())
    }

    /// Reverse one axis without copying; obtain a writable view of the same
    /// backing allocation. Requires exclusive borrowing of the Tensor.
    pub fn flip_view_mut(&mut self, axis: usize) -> Result<SignedTensorViewMut<'_, R>, TensorError> {
        self.signed_view_mut()?.flip_mut(axis)
    }

    /// Checked, zero-copy mutable transpose directly from the canonical Tensor.
    pub fn transpose_view_mut(&mut self, a: usize, b: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.transpose_mut(a, b)
    }

    /// Checked, zero-copy mutable axis move directly from Tensor.
    pub fn movedim_view_mut(&mut self, from: usize, to: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.movedim_mut(from, to)
    }

    /// Checked, zero-copy mutable axis range directly from Tensor.
    pub fn narrow_view_mut(&mut self, axis: usize, start: usize, length: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.narrow_mut(axis, start, length)
    }

    /// Merge an affine axis group without copying, directly from Tensor.
    pub fn flatten_view_mut<const OUT: usize>(&mut self, start: usize, end: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.flatten_view_mut::<OUT>(start, end)
    }

    /// Zero-copy mutable swap of two axes on the primary Tensor API.
    pub fn swapaxes_view_mut(&mut self, a: usize, b: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.swapaxes_mut(a, b)
    }

    /// Mutable singleton-axis insertion without allocating or copying.
    pub fn unsqueeze_view_mut<const OUT: usize>(&mut self, axis: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.unsqueeze_mut::<OUT>(axis)
    }

    /// Mutable singleton-axis removal without allocating or copying.
    pub fn squeeze_axis_view_mut<const OUT: usize>(&mut self, axis: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.squeeze_axis_mut::<OUT>(axis)
    }

    /// Mutable forward-contiguous zero-copy reshape directly on Tensor.
    pub fn reshape_view_mut<const OUT: usize>(&mut self, dims: [usize; OUT])
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.reshape_view_mut::<OUT>(dims)
    }

    /// Tensor-first consuming signed slice while its exclusive mutable borrow
    /// remains alive. Returns one validated, injective mutable view only.
    #[inline]
    pub fn slice_axis_signed_mut(&mut self, axis: usize, spec: SignedSliceSpec)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.slice_axis_mut(axis, spec)
    }

    /// Tensor-first writable permutation: the temporary view is moved, never
    /// cloned, and the entire physical mapping is revalidated for injectivity.
    #[inline]
    pub fn permute_signed_mut(&mut self, axes: [usize; R])
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.permute_mut(axes)
    }

    /// Reverse exactly one axis while preserving the sole mutable borrow.
    /// The checked signed view constructor revalidates bounds and injectivity.
    #[inline]
    pub fn flip_signed_mut(&mut self, axis: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.flip_mut(axis)
    }

    /// Reorder axes of the exclusive Tensor without copying payload values.
    #[inline]
    pub fn transpose_signed_mut(&mut self, a: usize, b: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.transpose_mut(a, b)
    }

    #[inline]
    pub fn swapaxes_signed_mut(&mut self, a: usize, b: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.swapaxes_mut(a, b)
    }

    #[inline]
    pub fn movedim_signed_mut(&mut self, from: usize, to: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.movedim_mut(from, to)
    }

    #[inline]
    pub fn narrow_signed_mut(&mut self, axis: usize, start: usize, length: usize)
        -> Result<SignedTensorViewMut<'_, R>, TensorError>
    {
        self.signed_view_mut()?.narrow_mut(axis, start, length)
    }

    #[inline]
    pub fn flatten_signed_mut<const OUT: usize>(&mut self, start: usize, end: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.flatten_view_mut::<OUT>(start, end)
    }

    #[inline]
    pub fn reshape_signed_mut<const OUT: usize>(&mut self, dims: [usize; OUT])
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.reshape_view_mut::<OUT>(dims)
    }

    #[inline]
    pub fn unsqueeze_signed_mut<const OUT: usize>(&mut self, axis: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.unsqueeze_mut::<OUT>(axis)
    }

    #[inline]
    pub fn squeeze_axis_signed_mut<const OUT: usize>(&mut self, axis: usize)
        -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        self.signed_view_mut()?.squeeze_axis_mut::<OUT>(axis)
    }

    /// A signed-stride mutable view is available only after a conservative
    /// proof that logical indices never alias one another physically.
    pub fn as_strided_signed_mut<const OUT: usize>(
        &mut self, dims: [usize; OUT], strides: [isize; OUT], origin: isize,
    ) -> Result<SignedTensorViewMut<'_, OUT>, TensorError>
    where Rank<OUT>: ValidRank,
    {
        SignedTensorViewMut::new(dims, strides, origin, self.as_mut_slice())
    }
}
