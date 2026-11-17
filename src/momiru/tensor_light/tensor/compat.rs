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
// Subcomponent:  Compat
// Module:        momiru::tensor_light::tensor::compat
// Source:        src/tensor_light/tensor/compat.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Compatibility adapters for fixed-rank tensor metadata and borrowing APIs.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Provide nonbreaking names for validated shape and borrow access.
//   - Check coordinate-slice rank and bounds before producing references.
//   - Preserve legacy scalar-value access methods and their semantics.
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

//! Non-breaking naming adapters for fixed-rank f32 tensor metadata and borrows.
//!
//! Coordinate slices validate rank and bounds once. Existing get([usize;R])
//! continues returning a copy; `try_get(&[usize])` borrows as in Generic ND.
use super::{Rank, Tensor, TensorElement, TensorError, ValidRank};

impl<T: TensorElement, const R: usize> Tensor<T, R>
where Rank<R>: ValidRank,
{
    /// Compile-time rank, exposed under the same name as Generic ND.
    #[inline(always)] pub const fn rank(&self) -> usize { R }
    /// No allocation: fixed-rank metadata iterator, matching Generic ND.
    #[inline] pub fn dimensions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.dims().iter().copied()
    }
    #[inline] pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims().get(axis).copied()
    }
    #[inline] pub fn size(&self, axis: usize) -> Option<usize> { self.dim(axis) }
    #[inline] pub fn is_contiguous(&self) -> bool { self.is_c_contiguous() }
    #[inline] pub fn is_fortran_contiguous(&self) -> bool { self.is_f_contiguous() }
    #[inline] pub fn is_unique_certified(&self) -> bool { self.is_unique() }
    #[inline] pub fn is_exhaustive_certified(&self) -> bool { self.is_exhaustive() }

    /// Constant-time physical borrow of the logical C-order contiguous data.
    /// Owning fixed-rank f32 tensors are always contiguous.
    #[inline(always)] pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        Some(self.as_slice())
    }
    /// No metadata allocation, offset arithmetic or layout traversal.
    #[inline(always)] pub fn as_contiguous_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(self.as_mut_slice())
    }
    /// Fast-path flattened reference lookup. Unlike N-D get, no rank-axes loop.
    #[inline(always)] pub fn get_linear(&self, linear: usize) -> Option<&T> {
        self.as_slice().get(linear)
    }
    /// Fast-path flattened mutable lookup, still checked for bounds.
    #[inline(always)] pub fn get_linear_mut(&mut self, linear: usize) -> Option<&mut T> {
        self.as_mut_slice().get_mut(linear)
    }
    /// Zero-copy extraction only when the tensor actually adopted a Vec.
    /// Aligned non-Vec allocations are NOT silently copied.
    #[inline] pub fn try_into_contiguous_vec(self) -> Result<Vec<T>, Self> {
        self.try_into_vec()
    }
    /// Explicit host-only copy alias; no plan-free numerical computation.
    #[inline] pub fn copy_from_host(&mut self, src: &Self) -> Result<(), TensorError> {
        self.copy_from_tensor(src)
    }
    #[inline] pub fn copy_from_host_copy(&mut self, src: &Self) -> Result<(), TensorError> {
        self.copy_from_tensor(src)
    }
    #[inline] pub fn try_fill_in_place(&mut self, value: T) -> Result<(), TensorError> {
        self.fill_in_place(value);
        Ok(())
    }

    /// Validated reference-returning coordinate-slice load.
    #[inline]
    pub fn try_get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        let coords: [usize; R] = indices.try_into()
            .map_err(|_| TensorError::IndexOutOfBounds)?;
        self.try_get_ref(coords)
    }

    /// Multiple mutable host borrows only at distinct physical offsets.
    /// For N=0 returns an empty array; for N>0 validates every position.
    /// This host-only indexing helper does not bypass plan-bound arithmetic.
    pub fn get_disjoint_mut<const N: usize>(
        &mut self, indices: [&[usize]; N]
    ) -> Result<[&mut T; N], TensorError> {
        let mut positions = [0usize; N];
        for (offset, coordinates) in positions.iter_mut().zip(indices) {
            let coords: [usize; R] = coordinates.try_into()
                .map_err(|_| TensorError::IndexOutOfBounds)?;
            *offset = self.shape().checked_offset(coords)
                .ok_or(TensorError::IndexOutOfBounds)?;
        }
        self.as_mut_slice().get_disjoint_mut(positions).map_err(|error| match error {
            core::slice::GetDisjointMutError::IndexOutOfBounds => TensorError::IndexOutOfBounds,
            core::slice::GetDisjointMutError::OverlappingIndices => TensorError::OverlappingMutableView,
        })
    }
}
