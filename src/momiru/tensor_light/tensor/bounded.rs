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
// Subcomponent:  Bounded
// Module:        momiru::tensor_light::tensor::bounded
// Source:        src/tensor_light/tensor/bounded.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Budget-checked f32 tensor constructors that reuse validated fixed-rank
//   creation plans.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Enforce user-specified logical work quotas before side effects.
//   - Reuse admitted creation metadata to avoid repeated shape traversal.
//   - Keep rank and element-type restrictions intact.
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

//! Explicit logical-work quotas for fixed-rank f32 constructors.
//!
//! The budget is checked only once at the allocation boundary; admitted
//! constructors reuse the checked creation plan and avoid a redundant shape
//! traversal. These helpers neither generalize the f32 dtype nor allow rank 9.
use super::{AlignedBuffer, CreationPlan, Rank, Tensor, TensorElement, TensorError, UninitTensor, ValidRank};

#[inline]
fn with_budget<const R: usize>(
    dims: [usize; R], limit: usize,
) -> Result<CreationPlan<R>, TensorError>
where Rank<R>: ValidRank {
    let plan = CreationPlan::new(dims)?;
    if plan.len() > limit { return Err(TensorError::ResourceInsufficient); }
    Ok(plan)
}

impl<T: TensorElement, const R: usize> Tensor<T, R>
where Rank<R>: ValidRank {
    /// Allocate f32 zeros only when the logical element count fits the limit.
    #[inline]
    pub fn zeros_bounded(dims: [usize; R], limit: usize) -> Result<Self, TensorError> {
        Self::zeros_with_plan(with_budget(dims, limit)?)
    }
    #[inline]
    pub fn full_bounded(
        dims: [usize; R], limit: usize, value: T,
    ) -> Result<Self, TensorError> {
        Self::full_with_plan(with_budget(dims, limit)?, value)
    }
    #[inline]
    pub fn new_bounded(
        dims: [usize; R], limit: usize, value: T,
    ) -> Result<Self, TensorError> {
        Self::full_bounded(dims, limit, value)
    }
    /// Checked uninitialized allocation; never permits reading before init.
    #[inline]
    pub fn empty_bounded(
        dims: [usize; R], limit: usize,
    ) -> Result<UninitTensor<T, R>, TensorError> {
        UninitTensor::from_plan(with_budget(dims, limit)?)
    }
    /// Validate budget and length before copying the first element.
    pub fn from_slice_bounded(
        dims: [usize; R], values: &[T], limit: usize,
    ) -> Result<Self, TensorError> {
        let plan = with_budget(dims, limit)?;
        if plan.len() != values.len() { return Err(TensorError::LengthMismatch); }
        // Avoid a second C-order shape computation: the plan was validated
        // and the input length was checked before allocating the copy.
        let data = AlignedBuffer::from_slice_aligned(values, plan.alignment())?;
        Ok(Self { shape: *plan.shape(), data })
    }
    /// Fallible, exact-length iterator ingestion; no intermediate copy and
    /// no iterator consumption when the element budget is exceeded.
    pub fn from_iter_exact_bounded(
        dims: [usize; R], limit: usize, items: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorError> {
        let plan = with_budget(dims, limit)?;
        let count = plan.len();
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| TensorError::AllocationFailed)?;
        let mut iterator = items.into_iter();
        for _ in 0..count {
            values.push(iterator.next().ok_or(TensorError::LengthMismatch)?);
        }
        if iterator.next().is_some() { return Err(TensorError::LengthMismatch); }
        // Adopt the just-built Vec, preserving its actual allocation alignment.
        Ok(Self { shape: *plan.shape(), data: AlignedBuffer::from_vec(values) })
    }
    #[inline]
    pub fn from_fn_bounded<F>(
        dims: [usize; R], limit: usize, make: F,
    ) -> Result<Self, TensorError>
    where F: FnMut([usize; R]) -> T {
        Self::from_fn_with_plan(with_budget(dims, limit)?, make)
    }
    /// Direct linear initialization: no per-element N-D coordinate updates.
    #[inline]
    pub fn from_elements_bounded<F>(
        dims: [usize; R], limit: usize, mut make: F,
    ) -> Result<Self, TensorError>
    where F: FnMut() -> T {
        Self::from_linear_fn_with_plan(with_budget(dims, limit)?, |_| make())
    }
    #[inline]
    pub fn zeros_like_bounded(&self, limit: usize) -> Result<Self, TensorError> {
        Self::zeros_with_plan(self.bounded_like_plan(limit)?)
    }
    #[inline]
    pub fn full_like_bounded(&self, limit: usize, value: T)
        -> Result<Self, TensorError> {
        Self::full_with_plan(self.bounded_like_plan(limit)?, value)
    }
    #[inline]
    pub fn from_fn_like_bounded<F>(&self, limit: usize, make: F)
        -> Result<Self, TensorError>
    where F: FnMut([usize; R]) -> T {
        Self::from_fn_with_plan(self.bounded_like_plan(limit)?, make)
    }
    #[inline]
    pub fn from_elements_like_bounded<F>(&self, limit: usize, mut make: F)
        -> Result<Self, TensorError>
    where F: FnMut() -> T {
        Self::from_linear_fn_with_plan(self.bounded_like_plan(limit)?, |_| make())
    }
    /// Input/Output shape guards precede all data writes.
    #[inline]
    pub fn copy_from_host_bounded(
        &mut self, source: &Self, limit: usize,
    ) -> Result<(), TensorError> {
        if self.len() > limit { return Err(TensorError::ResourceInsufficient); }
        self.copy_from_tensor(source)
    }
    #[inline]
    pub fn try_fill_host_bounded(&mut self, limit: usize, value: T)
        -> Result<(), TensorError> {
        if self.len() > limit { return Err(TensorError::ResourceInsufficient); }
        self.fill_in_place(value);
        Ok(())
    }
    #[inline]
    fn bounded_like_plan(&self, limit: usize) -> Result<CreationPlan<R>, TensorError> {
        let plan=CreationPlan::with_alignment(*self.dims(),self.alignment())?;
        if plan.len()>limit { return Err(TensorError::ResourceInsufficient); }
        Ok(plan)
    }
}

impl<const R: usize> Tensor<f32, R>
where Rank<R>: ValidRank {
    #[inline]
    pub fn ones_bounded(dims: [usize; R], limit: usize) -> Result<Self, TensorError> {
        Self::full_with_plan(with_budget(dims, limit)?, 1.0)
    }
    #[inline]
    pub fn ones_like_bounded(&self, limit: usize) -> Result<Self, TensorError> {
        Self::full_with_plan(self.bounded_like_plan(limit)?, 1.0)
    }
}
