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
// Subcomponent:  Auxiliaries
// Module:        momiru::tensor_light::tensor::auxiliaries
// Source:        src/tensor_light/tensor/auxiliaries.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Narrowly typed host-side result containers for index and predicate outputs
//   from completed UASDEA plans.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Represent canonical u64 index-result payloads.
//   - Represent explicitly unpacked u8 predicate-result payloads.
//   - Prevent auxiliary result types from becoming arbitrary non-f32 tensors.
//   - Avoid implicit computation or transfer when results are inspected.
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

//! Separate, narrowly typed host-side auxiliary result records.
//!
//! These are **not** general-purpose non-f32 Tensor types. They represent
//! the canonical u64 index-result ABI and an explicitly unpacked u8 predicate
//! output after a completed frozen graph. No computation or device transfer
//! is initiated when these records are inspected.

use super::{Rank, Shape, TensorError, ValidRank};

#[inline]
fn linear_index<const R: usize>(shape: &Shape<R>, index: [usize; R]) -> Result<usize, TensorError>
where
    Rank<R>: ValidRank,
{
    // Shape construction already proved every reachable f32 offset fits the
    // validated span. Reuse its checked-index boundary rather than repeating
    // per-axis checked arithmetic for each completed auxiliary scalar read.
    shape.checked_offset(index).ok_or(TensorError::IndexOutOfBounds)
}

/// Host-resident auxiliary `u64` indices from an already-completed result.
/// `IndexResult` does not implement the public f32 `Tensor` element trait.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexResult<const R: usize>
where
    Rank<R>: ValidRank,
{
    shape: Shape<R>,
    values: Vec<u64>,
}

impl<const R: usize> IndexResult<R>
where
    Rank<R>: ValidRank,
{
    /// Consume an already-verified host index output without copying.
    /// Index results may not be read while an AOT kernel is in flight.
    pub fn try_from_vec(dims: [usize; R], values: Vec<u64>) -> Result<Self, TensorError> {
        let shape = Shape::<R>::contiguous(dims)?;
        if shape.len() != values.len() { return Err(TensorError::LengthMismatch); }
        shape.len().checked_mul(core::mem::size_of::<u64>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(TensorError::AllocationOverflow)?;
        Ok(Self { shape, values })
    }

    #[inline(always)]
    pub const fn shape(&self) -> &Shape<R> { &self.shape }
    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.shape.len() }
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.shape.len() }
    /// Actual bytes of the u64 INDEX_RESULT payload, NOT the f32-only
    /// `Shape::nbytes()` value. Full u64 capacity was checked at construction.
    #[inline(always)]
    pub const fn nbytes(&self) -> usize {
        self.shape.len() * core::mem::size_of::<u64>()
    }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool { self.shape.is_empty() }
    #[inline(always)]
    pub fn as_slice(&self) -> &[u64] { &self.values }
    #[inline(always)]
    pub fn into_vec(self) -> Vec<u64> { self.values }

    pub fn get(&self, index: [usize; R]) -> Result<u64, TensorError> {
        let offset = linear_index(&self.shape, index)?;
        self.values.get(offset).copied().ok_or(TensorError::IndexOutOfBounds)
    }

    /// Host `usize` width is not assumed to equal the canonical `u64` wire
    /// index width: narrow only after a checked conversion.
    pub fn try_get_usize(&self, index: [usize; R]) -> Result<usize, TensorError> {
        usize::try_from(self.get(index)?).map_err(|_| TensorError::IndexOutOfBounds)
    }
}

/// One explicitly unpacked u8 predicate representation. Only 0/1 is legal.
/// Offline may choose another mask representation; it must be decoded by a
/// frozen conversion before producing this host auxiliary record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicateMask<const R: usize>
where
    Rank<R>: ValidRank,
{
    shape: Shape<R>,
    values: Vec<u8>,
}

impl<const R: usize> PredicateMask<R>
where
    Rank<R>: ValidRank,
{
    pub fn try_from_vec(dims: [usize; R], values: Vec<u8>) -> Result<Self, TensorError> {
        let shape = Shape::<R>::contiguous(dims)?;
        if shape.len() != values.len() { return Err(TensorError::LengthMismatch); }
        // Even though the f32 shape budget is currently stricter, validate
        // the actual u8 payload contract independently of that implementation
        // detail. This remains valid if the auxiliary geometry is later split.
        shape.len().checked_mul(core::mem::size_of::<u8>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(TensorError::AllocationOverflow)?;
        if values.iter().any(|&b| b > 1) { return Err(TensorError::InvalidMaskValue); }
        Ok(Self { shape, values })
    }

    #[inline(always)]
    pub const fn shape(&self) -> &Shape<R> { &self.shape }
    #[inline(always)]
    pub const fn ndim(&self) -> usize { R }
    #[inline(always)]
    pub const fn len(&self) -> usize { self.shape.len() }
    #[inline(always)]
    pub const fn numel(&self) -> usize { self.shape.len() }
    /// Actual bytes of the unpacked u8 PREDICATE_MASK payload. A packed mask
    /// is a different auxiliary representation and must be decoded before this
    /// host type is constructed; this is NOT the f32 `Shape::nbytes()` count.
    #[inline(always)]
    pub const fn nbytes(&self) -> usize { self.shape.len() }
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] { &self.values }
    #[inline(always)]
    pub fn into_vec(self) -> Vec<u8> { self.values }
    pub fn get(&self, index: [usize; R]) -> Result<bool, TensorError> {
        let offset = linear_index(&self.shape, index)?;
        self.values.get(offset).map(|&b| b != 0).ok_or(TensorError::IndexOutOfBounds)
    }
}
