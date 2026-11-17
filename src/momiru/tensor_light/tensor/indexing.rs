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
// Subcomponent:  Indexing
// Module:        momiru::tensor_light::tensor::indexing
// Source:        src/tensor_light/tensor/indexing.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Checked and trait-based scalar f32 tensor indexing, including signed-index
//   access.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Implement checked coordinate and scalar get/set operations.
//   - Preserve Index and IndexMut behavior for supported host data access.
//   - Do not introduce eager gather, scatter or bulk graph computation.
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

//! Scalar indexing for the concrete F32 tensor core.
//!
//! `Index` / `IndexMut`, checked accessors, signed indexing, and scalar
//! get/set operations remain host-side metadata/access utilities. Bulk gather,
//! scatter, masking, and selection execute only through frozen UASDEA plans.

use core::ops::{Index, IndexMut};

use super::{
    normalize_index, Rank, Shape, Tensor, TensorElement, TensorError, TensorView, TensorViewMut,
    ValidRank,
};

impl<const R: usize> Shape<R>
where
    Rank<R>: ValidRank,
{
    #[inline(always)]
    pub fn checked_offset(&self, index: [usize; R]) -> Option<usize> {
        // Short-circuit before *any* multiply: empty shapes can legally have
        // huge dimensions/strides that overflow on an unrealizable coordinate.
        if self.is_empty() { return None; }
        let mut offset = 0usize;
        let mut axis = 0usize;
        while axis < R {
            if index[axis] >= self.dims()[axis] {
                return None;
            }
            // `Shape::strided` validates that the maximum in-bounds offset fits
            // in `usize`, so checked arithmetic here would repeat work on every
            // hot scalar access without adding safety.
            offset += index[axis] * self.strides()[axis];
            axis += 1;
        }
        Some(offset)
    }

    #[inline(always)]
    pub fn checked_offset_signed(&self, index: [isize; R]) -> Option<usize> {
        if self.is_empty() { return None; }
        let mut offset = 0usize;
        let mut axis = 0usize;
        while axis < R {
            let coordinate = normalize_index(index[axis], self.dims()[axis]).ok()?;
            offset += coordinate * self.strides()[axis];
            axis += 1;
        }
        Some(offset)
    }

    /// # Safety
    /// Every coordinate must be in bounds and the resulting offset must fit the
    /// backing span described by this shape.
    #[inline(always)]
    pub unsafe fn offset_unchecked(&self, index: [usize; R]) -> usize {
        let mut offset = 0usize;
        let mut axis = 0usize;
        while axis < R {
            offset += index[axis] * self.strides()[axis];
            axis += 1;
        }
        offset
    }
}

impl<T: TensorElement, const R: usize> Tensor<T, R>
where
    Rank<R>: ValidRank,
{
    /// Checked scalar load. Rank and bounds are validated once at the API boundary.
    #[inline(always)]
    pub fn get(&self, index: [usize; R]) -> Result<T, TensorError> {
        Ok(*self.try_get_ref(index)?)
    }

    /// Safe reference to a Ready, exclusively owned host tensor's element.
    #[inline(always)]
    pub fn try_get_ref(&self, index: [usize; R]) -> Result<&T, TensorError> {
        let offset = self.shape.checked_offset(index).ok_or(TensorError::IndexOutOfBounds)?;
        Ok(unsafe { self.data.as_slice().get_unchecked(offset) })
    }

    #[inline(always)]
    pub fn try_get_mut(&mut self, index: [usize; R]) -> Result<&mut T, TensorError> {
        let offset = self.shape.checked_offset(index).ok_or(TensorError::IndexOutOfBounds)?;
        Ok(unsafe { self.data.as_mut_slice().get_unchecked_mut(offset) })
    }

    #[inline(always)]
    pub fn get_signed(&self, index: [isize; R]) -> Option<&T> {
        let offset = self.shape.checked_offset_signed(index)?;
        Some(unsafe { self.data.as_slice().get_unchecked(offset) })
    }

    #[inline(always)]
    pub fn get_signed_mut(&mut self, index: [isize; R]) -> Option<&mut T> {
        let offset = self.shape.checked_offset_signed(index)?;
        Some(unsafe { self.data.as_mut_slice().get_unchecked_mut(offset) })
    }

    #[inline(always)]
    pub fn set(&mut self, index: [usize; R], value: T) -> Result<(), TensorError> {
        let slot = self.try_get_mut(index)?;
        *slot = value;
        Ok(())
    }

    #[inline(always)]
    pub fn set_signed(&mut self, index: [isize; R], value: T) -> Result<(), TensorError> {
        let slot = self
            .get_signed_mut(index)
            .ok_or(TensorError::IndexOutOfBounds)?;
        *slot = value;
        Ok(())
    }

    /// # Safety
    /// `index` must be in bounds for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; R]) -> &T {
        let offset = unsafe { self.shape.offset_unchecked(index) };
        unsafe { self.data.as_slice().get_unchecked(offset) }
    }

    /// # Safety
    /// `index` must be in bounds for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; R]) -> &mut T {
        let offset = unsafe { self.shape.offset_unchecked(index) };
        unsafe { self.data.as_mut_slice().get_unchecked_mut(offset) }
    }

    /// # Safety
    /// `index` must be in bounds for every axis.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: [usize; R], value: T) {
        let slot = unsafe { self.get_unchecked_mut(index) };
        *slot = value;
    }
}

impl<'a, T: TensorElement, const R: usize> TensorView<'a, T, R>
where
    Rank<R>: ValidRank,
{
    #[inline(always)]
    pub fn get(&self, index: [usize; R]) -> Option<&T> {
        let offset = self.shape.checked_offset(index)?;
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    #[inline(always)]
    pub fn get_signed(&self, index: [isize; R]) -> Option<&T> {
        let offset = self.shape.checked_offset_signed(index)?;
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    /// # Safety
    /// `index` must be in bounds for every axis.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; R]) -> &T {
        let offset = unsafe { self.shape.offset_unchecked(index) };
        unsafe { self.data.get_unchecked(offset) }
    }
}

impl<'a, T: TensorElement, const R: usize> TensorViewMut<'a, T, R>
where
    Rank<R>: ValidRank,
{
    #[inline(always)]
    pub fn get(&self, index: [usize; R]) -> Option<&T> {
        let offset = self.shape.checked_offset(index)?;
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    #[inline(always)]
    pub fn get_mut(&mut self, index: [usize; R]) -> Option<&mut T> {
        let offset = self.shape.checked_offset(index)?;
        Some(unsafe { self.data.get_unchecked_mut(offset) })
    }

    #[inline(always)]
    pub fn get_signed(&self, index: [isize; R]) -> Option<&T> {
        let offset = self.shape.checked_offset_signed(index)?;
        Some(unsafe { self.data.get_unchecked(offset) })
    }

    #[inline(always)]
    pub fn get_signed_mut(&mut self, index: [isize; R]) -> Option<&mut T> {
        let offset = self.shape.checked_offset_signed(index)?;
        Some(unsafe { self.data.get_unchecked_mut(offset) })
    }

    #[inline(always)]
    pub fn set(&mut self, index: [usize; R], value: T) -> Result<(), TensorError> {
        let slot = self.get_mut(index).ok_or(TensorError::IndexOutOfBounds)?;
        *slot = value;
        Ok(())
    }

    #[inline(always)]
    pub fn set_signed(&mut self, index: [isize; R], value: T) -> Result<(), TensorError> {
        let slot = self
            .get_signed_mut(index)
            .ok_or(TensorError::IndexOutOfBounds)?;
        *slot = value;
        Ok(())
    }
}

impl<const R: usize> Tensor<f32, R>
where
    Rank<R>: ValidRank,
{
    /// Returns the only element of a single-element tensor.
    pub fn item(&self) -> Result<f32, TensorError> {
        if self.len() != 1 {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(self.as_slice()[0])
    }

    #[inline]
    pub fn at(&self, index: [isize; R]) -> Result<f32, TensorError> {
        self.get_signed(index)
            .copied()
            .ok_or(TensorError::IndexOutOfBounds)
    }

    #[inline]
    pub fn set_at(&mut self, index: [isize; R], value: f32) -> Result<(), TensorError> {
        self.set_signed(index, value)
    }

}

impl<T: TensorElement, const R: usize> Index<[usize; R]> for Tensor<T, R>
where
    Rank<R>: ValidRank,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.try_get_ref(index).expect("tensor index out of bounds")
    }
}

impl<T: TensorElement, const R: usize> IndexMut<[usize; R]> for Tensor<T, R>
where
    Rank<R>: ValidRank,
{
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
        self.try_get_mut(index).expect("tensor index out of bounds")
    }
}

impl<'a, T: TensorElement, const R: usize> Index<[usize; R]> for TensorView<'a, T, R>
where
    Rank<R>: ValidRank,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index).expect("tensor view index out of bounds")
    }
}

impl<'a, T: TensorElement, const R: usize> Index<[usize; R]> for TensorViewMut<'a, T, R>
where
    Rank<R>: ValidRank,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index).expect("tensor view index out of bounds")
    }
}

impl<'a, T: TensorElement, const R: usize> IndexMut<[usize; R]> for TensorViewMut<'a, T, R>
where
    Rank<R>: ValidRank,
{
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
        self.get_mut(index).expect("tensor view index out of bounds")
    }
}

macro_rules! impl_tuple_index {
    ($rank:literal, $tuple:ty, [$($field:tt),+]) => {
        impl<T: TensorElement> Index<$tuple> for Tensor<T, $rank> {
            type Output = T;
            #[inline(always)]
            fn index(&self, index: $tuple) -> &Self::Output {
                &self[[$(index.$field),+]]
            }
        }

        impl<T: TensorElement> IndexMut<$tuple> for Tensor<T, $rank> {
            #[inline(always)]
            fn index_mut(&mut self, index: $tuple) -> &mut Self::Output {
                &mut self[[$(index.$field),+]]
            }
        }

        impl<'a, T: TensorElement> Index<$tuple> for TensorView<'a, T, $rank> {
            type Output = T;
            #[inline(always)]
            fn index(&self, index: $tuple) -> &Self::Output {
                &self[[$(index.$field),+]]
            }
        }

        impl<'a, T: TensorElement> Index<$tuple> for TensorViewMut<'a, T, $rank> {
            type Output = T;
            #[inline(always)]
            fn index(&self, index: $tuple) -> &Self::Output {
                &self[[$(index.$field),+]]
            }
        }

        impl<'a, T: TensorElement> IndexMut<$tuple> for TensorViewMut<'a, T, $rank> {
            #[inline(always)]
            fn index_mut(&mut self, index: $tuple) -> &mut Self::Output {
                &mut self[[$(index.$field),+]]
            }
        }
    };
}

impl_tuple_index!(1, (usize,), [0]);
impl_tuple_index!(2, (usize, usize), [0, 1]);
impl_tuple_index!(3, (usize, usize, usize), [0, 1, 2]);
impl_tuple_index!(4, (usize, usize, usize, usize), [0, 1, 2, 3]);
impl_tuple_index!(5, (usize, usize, usize, usize, usize), [0, 1, 2, 3, 4]);
impl_tuple_index!(6, (usize, usize, usize, usize, usize, usize), [0, 1, 2, 3, 4, 5]);
impl_tuple_index!(7, (usize, usize, usize, usize, usize, usize, usize), [0, 1, 2, 3, 4, 5, 6]);
impl_tuple_index!(
    8,
    (usize, usize, usize, usize, usize, usize, usize, usize),
    [0, 1, 2, 3, 4, 5, 6, 7]
);

impl<T: TensorElement> Index<()> for Tensor<T, 0> {
    type Output = T;
    #[inline(always)]
    fn index(&self, _index: ()) -> &Self::Output {
        &self[[]]
    }
}

impl<T: TensorElement> IndexMut<()> for Tensor<T, 0> {
    #[inline(always)]
    fn index_mut(&mut self, _index: ()) -> &mut Self::Output {
        &mut self[[]]
    }
}

impl<'a, T: TensorElement> Index<()> for TensorView<'a, T, 0> {
    type Output = T;
    #[inline(always)]
    fn index(&self, _index: ()) -> &Self::Output {
        &self[[]]
    }
}

impl<'a, T: TensorElement> Index<()> for TensorViewMut<'a, T, 0> {
    type Output = T;
    #[inline(always)]
    fn index(&self, _index: ()) -> &Self::Output {
        &self[[]]
    }
}

impl<'a, T: TensorElement> IndexMut<()> for TensorViewMut<'a, T, 0> {
    #[inline(always)]
    fn index_mut(&mut self, _index: ()) -> &mut Self::Output {
        &mut self[[]]
    }
}
