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
// Subcomponent:  Core and Storage
// Module:        momiru::tensor_light::tensor
// Source:        src/tensor_light/tensor.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Core f32 host tensor storage, allocation, indexing and zero-copy view
//   integration for fixed ranks 0 through 8.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Manage owned f32 data and compile-time shape metadata.
//   - Provide checked host construction, scalar access and view interfaces.
//   - Preserve zero-copy metadata operations without eager graph evaluation.
//   - Delegate numerical graph kernels to the frozen execution plan.
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

//! F32 tensor core for UASDEA.
//!
//! The concrete tensor dtype is fixed to F32 and rank is restricted to 0..=8.
//! Allocation/construction, host-buffer initialization, scalar access and
//! zero-copy metadata views are deliberately public on `Tensor`.
//! Computational kernels (including elementwise arithmetic and matrix multiply)
//! must execute through the same preselected UASDEA FrozenExecutionPlan; they are
//! not hidden eager methods or a second CPU-specific tensor execution system.

mod indexing;
mod view;
mod signed_view;
mod signed_view_mut;
mod axis;
mod numerics;
mod auxiliaries;
mod layout;
mod compat;
mod bounded;
pub use layout::Layout;

pub const MIN_RANK: usize = 0;
pub const MAX_RANK: usize = 8;
pub const DEFAULT_HOST_ALIGNMENT: usize = 64;

/// Guaranteed absence, definite duplication, or conservative uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalOverlap { None, Definite, Unknown }

mod dtype {
    //! Concrete F32 tensor dtype plus engine-only index/mask metadata dtypes.

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(u8)]
    pub enum DType {
        F32,
        /// Engine-only canonical index metadata. The concrete Tensor<T> API remains F32-only.
        I32,
        /// Engine-only canonical index metadata. The concrete Tensor<T> API remains F32-only.
        I64,
        /// Engine-only canonical mask/byte metadata. The concrete Tensor<T> API remains F32-only.
        U8,
        /// Auxiliary canonical u64 index result (NOT a public Tensor dtype).
        U64,
    }

    impl DType {
        #[inline(always)]
        pub const fn bit_width(self) -> usize {
            match self {
                Self::F32 | Self::I32 => 32,
                Self::I64 | Self::U64 => 64,
                Self::U8 => 8,
            }
        }

        #[inline(always)]
        pub const fn size_bytes(self) -> usize {
            match self {
                Self::F32 | Self::I32 => 4,
                Self::I64 | Self::U64 => 8,
                Self::U8 => 1,
            }
        }

        #[inline(always)]
        pub const fn is_subbyte(self) -> bool {
            false
        }

        #[inline(always)]
        pub fn packed_byte_count(self, elements: usize) -> Option<usize> {
            elements.checked_mul(self.size_bytes())
        }
    }

    mod sealed {
        pub trait Sealed {}
    }

    /// Sealed on purpose: the concrete tensor API of this variant accepts only f32.
    pub trait TensorElement: sealed::Sealed + Copy + Send + Sync + 'static {
        const DTYPE: DType;
    }

    impl sealed::Sealed for f32 {}
    impl TensorElement for f32 {
        const DTYPE: DType = DType::F32;
    }
}
pub use dtype::{DType, TensorElement};

mod error;
pub use error::TensorError;

mod rank {
    //! Compile-time rank contract restricted to 0..=8.

    pub struct Rank<const R: usize>;

    mod rank_sealed {
        pub trait SealedRank {}
        impl SealedRank for super::Rank<0> {}
        impl SealedRank for super::Rank<1> {}
        impl SealedRank for super::Rank<2> {}
        impl SealedRank for super::Rank<3> {}
        impl SealedRank for super::Rank<4> {}
        impl SealedRank for super::Rank<5> {}
        impl SealedRank for super::Rank<6> {}
        impl SealedRank for super::Rank<7> {}
        impl SealedRank for super::Rank<8> {}
    }

    /// Sealed rank contract: only Rank<0> through Rank<8> can implement it.
    pub trait ValidRank: rank_sealed::SealedRank {}
    impl ValidRank for Rank<0> {}
    impl ValidRank for Rank<1> {}
    impl ValidRank for Rank<2> {}
    impl ValidRank for Rank<3> {}
    impl ValidRank for Rank<4> {}
    impl ValidRank for Rank<5> {}
    impl ValidRank for Rank<6> {}
    impl ValidRank for Rank<7> {}
    impl ValidRank for Rank<8> {}
}
pub use rank::{Rank, ValidRank};

mod shape;
pub use shape::Shape;

/// Greatest common divisor for positive element strides. For two active
/// axes, their nonzero stride multiples collide if and only if the bounded
/// difference domain contains the minimal Diophantine solution.
#[inline]
fn stride_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact injectivity criterion for a map with exactly two non-singleton axes.
/// The sign of the strides does not change the set of possible index
/// differences; using unsigned magnitudes also handles isize::MIN safely.
#[inline]
fn exact_two_axis_injective(
    extent_a: usize, stride_a: usize, extent_b: usize, stride_b: usize,
) -> bool {
    if extent_a <= 1 || extent_b <= 1 { return true; }
    if stride_a == 0 || stride_b == 0 { return false; }
    let gcd = stride_gcd(stride_a, stride_b);
    (extent_a - 1) < stride_b / gcd || (extent_b - 1) < stride_a / gcd
}

/// A constructive, exact collision witness for any PAIR of active axes.
/// Other axes may be held fixed, so a pairwise collision suffices to prove
/// Definite overlap even for a rank-8 tensor.
#[inline]
fn mapping_has_definite_pair_collision<const R: usize>(
    dims: &[usize; R], strides: &[isize; R],
) -> bool {
    for a in 0..R {
        if dims[a] <= 1 { continue; }
        let sa = strides[a].unsigned_abs();
        if sa == 0 { return true; }
        for b in a + 1..R {
            if dims[b] <= 1 { continue; }
            let sb = strides[b].unsigned_abs();
            if !exact_two_axis_injective(dims[a], sa, dims[b], sb) {
                return true;
            }
        }
    }
    false
}

/// Constructive three-axis overlap witness that works for large tensors
/// without enumerating every logical element. For each triple, enumerate
/// bounded index DIFFERENCES on the two shortest axes and solve the third
/// difference exactly. A nonzero solution proves two distinct in-bounds
/// logical indices have the same storage offset, even with signed strides.
///
/// All arithmetic is host metadata only, has an explicit operation budget,
/// never creates a pointer or allocates heap memory, and is intentionally
/// incomplete: exhausting the budget must lead to Unknown, never None.
#[inline]
fn mapping_has_bounded_triple_collision<const R: usize>(
    dims: &[usize; R], strides: &[isize; R],
) -> bool {
    const MAX_CHECKS: usize = 4096;
    const MAX_CHECKS_PER_TRIPLE: usize = 256;
    let mut budget = MAX_CHECKS;
    for a in 0..R {
        if dims[a] <= 1 { continue; }
        for b in a + 1..R {
            if dims[b] <= 1 { continue; }
            for c in b + 1..R {
                if dims[c] <= 1 { continue; }
                // Sort just THREE axis identifiers by extent, on the stack.
                let mut axes = [a, b, c];
                if dims[axes[0]] > dims[axes[1]] { axes.swap(0, 1); }
                if dims[axes[1]] > dims[axes[2]] { axes.swap(1, 2); }
                if dims[axes[0]] > dims[axes[1]] { axes.swap(0, 1); }
                let delta_a = dims[axes[0]] - 1;
                let delta_b = dims[axes[1]] - 1;
                let delta_c = dims[axes[2]] - 1;
                let Some(width_a) = delta_a.checked_mul(2).and_then(|x| x.checked_add(1)) else {
                    continue;
                };
                let Some(width_b) = delta_b.checked_mul(2).and_then(|x| x.checked_add(1)) else {
                    continue;
                };
                let Some(checks) = width_a.checked_mul(width_b) else { continue; };
                if checks > MAX_CHECKS_PER_TRIPLE || checks > budget { continue; }
                budget -= checks;
                let Ok(delta_a) = i128::try_from(delta_a) else { continue; };
                let Ok(delta_b) = i128::try_from(delta_b) else { continue; };
                let Ok(delta_c) = i128::try_from(delta_c) else { continue; };
                let sa = strides[axes[0]] as i128;
                let sb = strides[axes[1]] as i128;
                let sc = strides[axes[2]] as i128;
                // Zero strides were handled before this helper is called.
                if sc == 0 { continue; }
                // A bounded delta (at most 127 for either enumerated axis)
                // times any isize stride cannot overflow i128; addition of
                // the two products also fits i128 on supported hosts.
                for da in -delta_a..=delta_a {
                    for db in -delta_b..=delta_b {
                        let required = -(da * sa + db * sb);
                        if required % sc == 0 {
                            let dc = required / sc;
                            if dc >= -delta_c && dc <= delta_c
                                && (da != 0 || db != 0 || dc != 0)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Sound pigeonhole certificate for multi-axis signed mappings.
///
/// Every logical element maps to one integral element offset inside a
/// bounding interval of `1 + sum((dim_i - 1) * abs(stride_i))` elements.
/// When the logical cardinality is strictly larger, two distinct logical
/// indices MUST refer to the same f32 element. This proof is O(R), uses no
/// heap memory and never creates a pointer. Overflow means *no proof*, not
/// permission to create a mutable view. Empty shapes and rank-zero scalars
/// cannot have an internal collision.
#[inline]
fn mapping_has_pigeonhole_collision<const R: usize>(
    dims: &[usize; R], strides: &[isize; R], len: usize,
) -> bool {
    if len <= 1 || dims.contains(&0) { return false; }
    let mut span = 1u128;
    for axis in 0..R {
        if dims[axis] <= 1 { continue; }
        let extent = (dims[axis] - 1) as u128;
        let stride = strides[axis].unsigned_abs() as u128;
        let Some(delta) = extent.checked_mul(stride) else { return false; };
        let Some(next) = span.checked_add(delta) else { return false; };
        span = next;
    }
    (len as u128) > span
}

/// Exact, allocation-free *tri-state* classification for bounded mappings.
///
/// Repeated offsets are a constructive proof of Definite overlap. Distinct
/// offsets prove None. A mapping outside this bounded proof or an overflow in
/// the mathematical address calculation is Unknown and must never authorize
/// a mutable reference. No payload pointer is ever created here.
#[inline]
fn bounded_exact_mapping_overlap<const R: usize>(
    dims: &[usize; R], strides: &[isize; R], len: usize,
) -> InternalOverlap {
    const LIMIT: usize = 128;
    if len <= 1 { return InternalOverlap::None; }
    if len > LIMIT || dims.contains(&0) { return InternalOverlap::Unknown; }

    let mut offsets = [0i128; LIMIT];
    for (linear, slot) in offsets[..len].iter_mut().enumerate() {
        let mut remaining = linear;
        let mut offset = 0i128;
        for axis in (0..R).rev() {
            let coordinate = remaining % dims[axis];
            remaining /= dims[axis];
            let Some(term) = (coordinate as i128).checked_mul(strides[axis] as i128) else {
                return InternalOverlap::Unknown;
            };
            let Some(sum) = offset.checked_add(term) else {
                return InternalOverlap::Unknown;
            };
            offset = sum;
        }
        if remaining != 0 { return InternalOverlap::Unknown; }
        *slot = offset;
    }
    offsets[..len].sort_unstable(); // in place; no heap allocation
    for index in 1..len {
        if offsets[index] == offsets[index - 1] {
            return InternalOverlap::Definite;
        }
    }
    InternalOverlap::None
}

/// One canonical rank-0..8 injectivity/overlap classifier for both signed and
/// nonnegative-stride metadata. It is exact for <=2 active axes and for every
/// valid mapping with <=128 elements. For larger complex patterns it proves
/// separated-stride uniqueness, pairwise collision and bounded triple-axis
/// collisions; other undecided cases remain Unknown rather than unsafe.
///
/// Crucially, the hot path for common contiguous/transposed layouts runs the
/// cheap sorted-stride certificate BEFORE any pairwise gcd or enumeration.
#[inline]
pub(crate) fn signed_mapping_overlap<const R: usize>(
    dims: &[usize; R], strides: &[isize; R], len: usize,
) -> InternalOverlap {
    if len <= 1 { return InternalOverlap::None; }
    let mut axes = [0usize; R];
    let mut count = 0usize;
    for axis in 0..R {
        if dims[axis] > 1 {
            if strides[axis] == 0 { return InternalOverlap::Definite; }
            axes[count] = axis;
            count += 1;
        }
    }
    // Common contiguous/strided case: prove separation without a gcd loop.
    for i in 1..count {
        let axis = axes[i];
        let mut j = i;
        while j > 0 && strides[axes[j - 1]].unsigned_abs() > strides[axis].unsigned_abs() {
            axes[j] = axes[j - 1];
            j -= 1;
        }
        axes[j] = axis;
    }
    let mut covered = 1u128;
    let mut separated = true;
    for &axis in &axes[..count] {
        let stride = strides[axis].unsigned_abs() as u128;
        if stride < covered { separated = false; break; }
        let extent = (dims[axis] - 1) as u128;
        let Some(delta) = extent.checked_mul(stride) else {
            separated = false;
            break;
        };
        let Some(next) = covered.checked_add(delta) else {
            separated = false;
            break;
        };
        covered = next;
    }
    if separated { return InternalOverlap::None; }

    // Exactly two active axes admit an O(log stride) Diophantine criterion,
    // with a proven Definite verdict when the bounded differences collide.
    if count == 2 {
        let a = axes[0];
        let b = axes[1];
        return if exact_two_axis_injective(
            dims[a], strides[a].unsigned_abs(), dims[b], strides[b].unsigned_abs(),
        ) { InternalOverlap::None } else { InternalOverlap::Definite };
    }
    // A fast cardinality-vs-physical-span (pigeonhole) witness detects
    // certain large multi-axis collisions missed by bounded enumeration.
    // It is a one-way proof: span >= numel NEVER proves injectivity.
    // Common contiguous/separated strides returned None above.
    if mapping_has_pigeonhole_collision(dims, strides, len) {
        return InternalOverlap::Definite;
    }
    // Constructive pair and bounded triple witnesses prove other collisions
    // without treating a mathematically undecided map as safely injective.
    if mapping_has_definite_pair_collision(dims, strides)
        || mapping_has_bounded_triple_collision(dims, strides)
    {
        return InternalOverlap::Definite;
    }
    bounded_exact_mapping_overlap(dims, strides, len)
}

#[inline]
pub(crate) fn signed_mapping_is_injective<const R: usize>(
    dims: &[usize; R], strides: &[isize; R], len: usize,
) -> bool {
    signed_mapping_overlap(dims, strides, len) == InternalOverlap::None
}
pub use auxiliaries::{IndexResult, PredicateMask};

/// Collapse a contiguous *logical axis group* into one affine stride without
/// touching payload memory. The enclosing tensor does NOT need to be contiguous.
/// Negative strides and all-zero-stride broadcasts are supported when the
/// selected axes have the same affine C-order flattening. This is exact, never
/// an optimistic alias or overlap assumption.
///
/// For empty groups, no address is reachable; shape cardinality becomes zero
/// before any potentially overflowing multiplication of the other extents.
pub(crate) fn collapse_flatten_group<const R: usize>(
    dims: &[usize; R], strides: &[isize; R], start: usize, end: usize,
) -> Result<(usize, isize), TensorError> {
    if start > end || end >= R { return Err(TensorError::InvalidRange); }
    if dims[start..=end].contains(&0) { return Ok((0, 0)); }

    let mut cardinality = 1usize;
    let mut inner_stride: Option<i128> = None;
    let mut scale = 1i128;
    for axis in (start..=end).rev() {
        let dim = dims[axis];
        if dim > 1 {
            if let Some(base_stride) = inner_stride {
                let expected = base_stride.checked_mul(scale)
                    .ok_or(TensorError::ShapeOverflow)?;
                if (strides[axis] as i128) != expected {
                    return Err(TensorError::NonContiguous);
                }
            } else {
                inner_stride = Some(strides[axis] as i128);
            }
        }
        cardinality = cardinality.checked_mul(dim).ok_or(TensorError::ShapeOverflow)?;
        scale = scale.checked_mul(dim as i128).ok_or(TensorError::ShapeOverflow)?;
    }
    let stride = inner_stride.unwrap_or(0);
    Ok((cardinality, isize::try_from(stride).map_err(|_| TensorError::ShapeOverflow)?))
}

mod storage {
    //! Aligned host storage for the concrete tensor core.

    use core::marker::PhantomData;
    use core::ptr::NonNull;
    use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};

    use super::{TensorElement, TensorError, DEFAULT_HOST_ALIGNMENT};

    pub struct AlignedBuffer<T: TensorElement> {
        pub(super) ptr: NonNull<T>,
        pub(super) len: usize,
        pub(super) alignment: usize,
        pub(super) layout: Option<Layout>,
        pub(super) vec_capacity: Option<usize>,
        pub(super) _marker: PhantomData<T>,
    }

    // SAFETY (Send): this is the unique owner of an allocation created by
    // std::alloc or adopted without copying from a Vec<T>. No safe method
    // constructs independent mutable owners. Moving ownership to a different
    // thread does not expose concurrent readers/writers, and both deallocation
    // strategies (Global/std::alloc) may run on the destination thread.
    // T is sealed by TensorElement (f32 only) and must itself be Send.
    unsafe impl<T: TensorElement> Send for AlignedBuffer<T> {}

    // SAFETY (Sync): shared access exposes only immutable slices/references,
    // while all safe mutable access requires &mut self. The internal pointer
    // cannot mutate through &self and no interior mutation/foreign writer is
    // installed by this host-only allocator. A raw pointer obtained from a
    // safe API may only be dereferenced under the caller's unsafe obligations.
    // T is sealed by TensorElement (f32 only) and must itself be Sync.
    unsafe impl<T: TensorElement> Sync for AlignedBuffer<T> {}

    /// A null result from the fallible raw allocator is a normal resource
    /// error, not a reason to terminate the process. The allocator-generated
    /// pointer is non-null on success and keeps its original provenance.
    /// Zero-size requests never reach this helper.
    #[inline]
    pub(super) fn checked_host_allocation<T>(raw: *mut u8) -> Result<NonNull<T>, TensorError> {
        NonNull::new(raw.cast::<T>()).ok_or(TensorError::AllocationFailed)
    }

    /// Empty Rust slices require a non-null and correctly aligned pointer.
    /// A synthetic pointer is never dereferenced; its integer address meets
    /// the published alignment rather than merely T's minimum alignment.
    #[inline]
    pub(super) fn aligned_empty_pointer<T>(alignment: usize) -> NonNull<T> {
        // SAFETY: alignment has been validated as a non-zero power of two.
        // The constructed pointer is only used for zero-length allocations.
        unsafe { NonNull::new_unchecked(core::ptr::without_provenance_mut::<T>(alignment)) }
    }

    impl<T: TensorElement> AlignedBuffer<T> {
        pub fn zeroed(len: usize) -> Result<Self, TensorError> {
            Self::zeroed_aligned(len, DEFAULT_HOST_ALIGNMENT)
        }

        pub fn zeroed_aligned(len: usize, alignment: usize) -> Result<Self, TensorError> {
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(TensorError::InvalidAlignment);
            }

            let effective_alignment = alignment.max(core::mem::align_of::<T>());
            if len == 0 {
                return Ok(Self {
                    ptr: aligned_empty_pointer(effective_alignment),
                    len: 0,
                    alignment: effective_alignment,
                    layout: None,
                    vec_capacity: None,
                    _marker: PhantomData,
                });
            }

            let bytes = core::mem::size_of::<T>()
                .checked_mul(len)
                .ok_or(TensorError::AllocationOverflow)?;
            let layout = Layout::from_size_align(bytes, effective_alignment)
                .map_err(|_| TensorError::AllocationOverflow)?;

            let raw = unsafe { alloc_zeroed(layout) };
            let ptr = checked_host_allocation::<T>(raw)?;
            Ok(Self {
                ptr,
                len,
                alignment: effective_alignment,
                layout: Some(layout),
                vec_capacity: None,
                _marker: PhantomData,
            })
        }

        pub fn from_slice(src: &[T]) -> Result<Self, TensorError> {
            Self::from_slice_aligned(src, DEFAULT_HOST_ALIGNMENT)
        }

        pub fn from_slice_aligned(src: &[T], alignment: usize) -> Result<Self, TensorError> {
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(TensorError::InvalidAlignment);
            }

            let effective_alignment = alignment.max(core::mem::align_of::<T>());
            if src.is_empty() {
                return Ok(Self {
                    ptr: aligned_empty_pointer(effective_alignment),
                    len: 0,
                    alignment: effective_alignment,
                    layout: None,
                    vec_capacity: None,
                    _marker: PhantomData,
                });
            }

            let bytes = core::mem::size_of::<T>()
                .checked_mul(src.len())
                .ok_or(TensorError::AllocationOverflow)?;
            let layout = Layout::from_size_align(bytes, effective_alignment)
                .map_err(|_| TensorError::AllocationOverflow)?;

            // Do not zero memory before an immediate full overwrite. This avoids an
            // unnecessary memset on tensor construction from existing data.
            let raw = unsafe { alloc(layout) };
            let ptr = checked_host_allocation::<T>(raw)?;
            unsafe {
                core::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), src.len());
            }

            Ok(Self {
                ptr,
                len: src.len(),
                alignment: effective_alignment,
                layout: Some(layout),
                vec_capacity: None,
                _marker: PhantomData,
            })
        }

        /// Adopt the original Vec allocation and its original deallocator.
        /// No memcpy, reallocation, padding, or fabricated SIMD alignment.
        pub fn from_vec(data: Vec<T>) -> Self {
            // Rust 1.99's safe Vec::into_parts retains the actual non-null
            // allocation pointer, length and capacity without memcpy or a
            // manually forgotten owner. Vec is sealed to f32 here.
            let (ptr, len, capacity) = data.into_parts();
            // Preserve the allocation's ACTUAL alignment. Never fabricate
            // 32/64-byte SIMD alignment for an adopted Vec.
            let pointer_alignment = 1usize << (ptr.as_ptr() as usize).trailing_zeros();
            Self {
                ptr, len,
                alignment: pointer_alignment.max(core::mem::align_of::<T>()),
                layout: None,
                vec_capacity: Some(capacity),
                _marker: PhantomData,
            }
        }

        #[inline(always)]
        pub const fn alignment(&self) -> usize {
            self.alignment
        }

        #[inline(always)]
        pub const fn len(&self) -> usize {
            self.len
        }

        #[inline(always)]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        #[inline(always)]
        pub fn as_ptr(&self) -> *const T {
            self.ptr.as_ptr()
        }

        #[inline(always)]
        pub fn as_mut_ptr(&mut self) -> *mut T {
            self.ptr.as_ptr()
        }

        #[inline(always)]
        pub fn as_slice(&self) -> &[T] {
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }

        #[inline(always)]
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    impl<T: TensorElement> Drop for AlignedBuffer<T> {
        fn drop(&mut self) {
            if let Some(capacity) = self.vec_capacity {
                // SAFETY: from_vec retained exactly the original Vec pointer,
                // length, capacity and allocation ownership.
                let _original_vec = unsafe {
                    Vec::from_parts(self.ptr, self.len, capacity)
                };
            } else if let Some(layout) = self.layout {
                unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
            }
        }
    }
}
pub(crate) use storage::AlignedBuffer;

mod owned {
    //! Owning tensor and explicitly uninitialized output storage.

    use core::marker::PhantomData;
    use core::ptr::NonNull;
    use std::alloc::{alloc, dealloc, Layout};

    use crate::tensor_light::tensor::view::{TensorView, TensorViewMut};
    use super::{
        AlignedBuffer, DType, Rank, Shape, TensorElement, TensorError, ValidRank,
        DEFAULT_HOST_ALIGNMENT,
    };

    pub struct Tensor<T: TensorElement, const R: usize>
    where
        Rank<R>: ValidRank,
    {
        pub(super) shape: Shape<R>,
        pub(super) data: AlignedBuffer<T>,
    }

    // Metadata-only Debug: does not print all potentially huge f32 payloads
    // or allocate a materialized tensor representation.
    impl<T: TensorElement, const R: usize> core::fmt::Debug for Tensor<T, R>
    where
        Rank<R>: ValidRank,
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Tensor")
                .field("dtype", &T::DTYPE)
                .field("shape", self.shape.dims())
                .field("alignment", &self.data.alignment())
                .finish_non_exhaustive()
        }
    }

    impl<T: TensorElement, const R: usize> Tensor<T, R>
    where
        Rank<R>: ValidRank,
    {
        /// Move a Vec<f32> into Tensor without copying or reallocating its data.
        /// Alignment remains the allocator's ACTUAL alignment, never a pretend 64B.
        pub fn try_from_vec(dims: [usize; R], values: Vec<T>) -> Result<Self, TensorError> {
            let shape = Shape::contiguous(dims)?;
            if shape.len() != values.len() { return Err(TensorError::LengthMismatch); }
            Ok(Self { shape, data: AlignedBuffer::from_vec(values) })
        }

        /// Canonical, zero-copy owner transfer from a Vec of f32 values.
        /// The source Vec's actual pointer alignment is retained, not upgraded.
        #[inline]
        pub fn from_vec(dims: [usize; R], values: Vec<T>) -> Result<Self, TensorError> {
            Self::try_from_vec(dims, values)
        }

        /// Copy an existing buffer into a separately owned contiguous Tensor.
        /// Host-side storage provisioning; no semantic graph operation is executed.
        /// The allocation preserves the current tensor's required alignment.
        #[inline]
        pub fn try_clone(&self) -> Result<Self, TensorError> {
            Self::from_slice_aligned(*self.dims(), self.as_slice(), self.alignment())
        }

        /// Export contiguous host data to a Vec with a checked, fallible allocation.
        /// Use try_into_vec for zero-copy recovery of an originally adopted Vec.
        pub fn try_to_vec(&self) -> Result<Vec<T>, TensorError> {
            let mut result = Vec::new();
            result.try_reserve_exact(self.len())
                .map_err(|_| TensorError::AllocationFailed)?;
            result.extend_from_slice(self.as_slice());
            Ok(result)
        }

        /// Provision the host buffer from equally sized initialized data.
        /// All shape/length checks happen before the first write.
        pub fn copy_from_slice(&mut self, src: &[T]) -> Result<(), TensorError> {
            if src.len() != self.len() { return Err(TensorError::LengthMismatch); }
            self.as_mut_slice().copy_from_slice(src);
            Ok(())
        }

        /// Exact-shape owned-host copy. No execution-plan search or computation.
        pub fn copy_from_tensor(&mut self, src: &Self) -> Result<(), TensorError> {
            if self.dims() != src.dims() { return Err(TensorError::ShapeMismatch); }
            self.as_mut_slice().copy_from_slice(src.as_slice());
            Ok(())
        }

        /// Initialize/overwrite every owned host element with one scalar value.
        /// This is storage provisioning, not a general eager tensor operation.
        #[inline]
        pub fn fill_in_place(&mut self, value: T) {
            self.as_mut_slice().fill(value);
        }

        /// Reinterpret only immutable shape metadata, moving the owned allocation
        /// without copying, reallocating, or changing the underlying address.
        /// If dimensions mismatch, return the original Tensor unmodified.
        pub fn try_reshape_owned<const OUT: usize>(
            self, dims: [usize; OUT],
        ) -> Result<Tensor<T, OUT>, (TensorError, Self)>
        where
            Rank<OUT>: ValidRank,
        {
            let new_shape = match Shape::<OUT>::contiguous(dims) {
                Ok(shape) => shape,
                Err(error) => return Err((error, self)),
            };
            if new_shape.len() != self.len() {
                return Err((TensorError::ShapeMismatch, self));
            }
            let Tensor { data, .. } = self;
            Ok(Tensor { shape: new_shape, data })
        }

        /// Recover the *original* Vec allocation, length, and capacity without
        /// a copy. Only Tensor values actually adopted from Vec qualify.
        /// Custom 64-byte allocations must never be reconstructed as Vec.
        /// On failure, the owning Tensor is returned unchanged.
        pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
            let Some(capacity) = self.data.vec_capacity else { return Err(self) };
            let owned = core::mem::ManuallyDrop::new(self);
            // SAFETY: AlignedBuffer::from_vec exclusively retained the
            // original Vec<T> pointer/length/capacity/allocator. ManuallyDrop
            // prevents double-free by AlignedBuffer::drop.
            Ok(unsafe { Vec::from_parts(owned.data.ptr, owned.data.len, capacity) })
        }

        #[inline(always)]
        pub const fn dtype(&self) -> DType { T::DTYPE }

        #[inline(always)]
        pub const fn shape(&self) -> &Shape<R> {
            &self.shape
        }

        #[inline(always)]
        pub const fn alignment(&self) -> usize { self.data.alignment() }

        #[inline(always)]
        pub const fn ndim(&self) -> usize { R }
        #[inline(always)]
        pub const fn is_scalar(&self) -> bool { R == 0 }
        #[inline(always)]
        pub fn signed_strides(&self) -> [isize; R] { self.shape.signed_strides() }
        #[inline(always)]
        pub const fn dims(&self) -> &[usize; R] { self.shape.dims() }
        #[inline(always)]
        pub const fn strides(&self) -> &[usize; R] { self.shape.strides() }
        #[inline(always)]
        pub const fn nbytes(&self) -> usize { self.shape.nbytes() }
        /// Logical f32 element count, distinct from physical span or Vec capacity.
        #[inline(always)]
        pub const fn numel(&self) -> usize { self.shape.numel() }
        /// Physical span of the owned host allocation in f32 elements.
        /// Dense owning Tensor always has span_len == numel.
        #[inline(always)]
        pub const fn span_len(&self) -> usize { self.shape.span_len() }
        #[inline(always)]
        pub fn is_c_contiguous(&self) -> bool { self.shape.is_c_contiguous() }
        #[inline(always)]
        pub fn is_f_contiguous(&self) -> bool { self.shape.is_f_contiguous() }
        #[inline(always)]
        pub const fn is_unique(&self) -> bool { true }
        #[inline(always)]
        pub const fn is_exhaustive(&self) -> bool { true }
        #[inline(always)]
        pub const fn internal_overlap(&self) -> super::InternalOverlap {
            super::InternalOverlap::None
        }

        #[inline(always)]
        pub fn as_slice(&self) -> &[T] {
            self.data.as_slice()
        }

        #[inline(always)]
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            self.data.as_mut_slice()
        }

        #[inline(always)]
        pub fn as_ptr(&self) -> *const T {
            self.data.as_ptr()
        }

        #[inline(always)]
        pub fn as_mut_ptr(&mut self) -> *mut T {
            self.data.as_mut_ptr()
        }

        #[inline(always)]
        pub const fn len(&self) -> usize {
            self.shape.len()
        }

        #[inline(always)]
        pub const fn is_empty(&self) -> bool {
            self.shape.is_empty()
        }

        /// Fallible-name companion to `view`, matching the documented public
        /// Tensor-first API. This *uniquely owned, initialized host* Tensor
        /// cannot be poisoned by the current host-only API; therefore no
        /// runtime state branch or extra allocation is necessary.
        #[inline(always)]
        pub fn try_view(&self) -> Result<TensorView<'_, T, R>, TensorError> {
            Ok(self.view())
        }

        /// Checked-name companion to `view_mut`. Rust's exclusive `&mut`
        /// borrow is the physical writer-exclusion certificate for this
        /// uniquely owned host allocation. No GPU/DMA writer is admitted here.
        #[inline(always)]
        pub fn try_view_mut(&mut self) -> Result<TensorViewMut<'_, T, R>, TensorError> {
            Ok(self.view_mut())
        }

        #[inline(always)]
        pub fn view(&self) -> TensorView<'_, T, R> {
            TensorView {
                shape: self.shape,
                data: self.data.as_slice(),
            }
        }

        #[inline(always)]
        pub fn view_mut(&mut self) -> TensorViewMut<'_, T, R> {
            TensorViewMut {
                shape: self.shape,
                data: self.data.as_mut_slice(),
            }
        }
    }

    /// Output tensor storage that is intentionally left uninitialized. This avoids a
    /// mandatory memset before an AOT kernel that is guaranteed to overwrite every
    /// element. It cannot be read until `assume_init` is called.
    pub struct UninitTensor<T: TensorElement, const R: usize>
    where
        Rank<R>: ValidRank,
    {
        shape: Shape<R>,
        ptr: NonNull<T>,
        len: usize,
        alignment: usize,
        layout: Option<Layout>,
        _marker: PhantomData<T>,
    }

    // Never read uninitialized payload in Debug/Display paths.
    impl<T: TensorElement, const R: usize> core::fmt::Debug for UninitTensor<T, R>
    where
        Rank<R>: ValidRank,
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("UninitTensor")
                .field("dtype", &T::DTYPE)
                .field("shape", self.shape.dims())
                .field("alignment", &self.alignment)
                .field("state", &"uninitialized")
                .finish_non_exhaustive()
        }
    }

    unsafe impl<T: TensorElement, const R: usize> Send for UninitTensor<T, R> where
        Rank<R>: ValidRank
    {
    }

    impl<T: TensorElement, const R: usize> UninitTensor<T, R>
    where
        Rank<R>: ValidRank,
    {
        pub fn new(dims: [usize; R]) -> Result<Self, TensorError> {
            Self::new_aligned(dims, DEFAULT_HOST_ALIGNMENT)
        }

        pub fn new_aligned(dims: [usize; R], alignment: usize) -> Result<Self, TensorError> {
            // The public path validates user-provided metadata once.
            let shape = Shape::contiguous(dims)?;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(TensorError::InvalidAlignment);
            }
            Self::allocate_validated_shape(shape, alignment)
        }

        /// Fast allocation path for a trusted, checked CreationPlan. The
        /// shape product and contiguous strides have already been validated;
        /// repeating Shape::contiguous for each output adds needless work,
        /// particularly for many small rank-8 host tensors.
        #[inline]
        fn allocate_validated_shape(shape: Shape<R>, alignment: usize) -> Result<Self, TensorError> {
            debug_assert!(alignment != 0 && alignment.is_power_of_two());
            let effective_alignment = alignment.max(core::mem::align_of::<T>());
            if shape.len() == 0 {
                return Ok(Self {
                    shape,
                    ptr: super::storage::aligned_empty_pointer(effective_alignment),
                    len: 0,
                    alignment: effective_alignment,
                    layout: None,
                    _marker: PhantomData,
                });
            }

            let len = shape.len();
            let bytes = core::mem::size_of::<T>()
                .checked_mul(len)
                .ok_or(TensorError::AllocationOverflow)?;
            let layout = Layout::from_size_align(bytes, effective_alignment)
                .map_err(|_| TensorError::AllocationOverflow)?;
            let raw = unsafe { alloc(layout) };
            let ptr = super::storage::checked_host_allocation::<T>(raw)?;

            Ok(Self {
                shape,
                ptr,
                len,
                alignment: effective_alignment,
                layout: Some(layout),
                _marker: PhantomData,
            })
        }

        #[inline(always)]
        pub const fn shape(&self) -> &Shape<R> {
            &self.shape
        }

        #[inline(always)]
        pub fn as_mut_ptr(&mut self) -> *mut T {
            self.ptr.as_ptr()
        }

        #[inline(always)]
        pub const fn alignment(&self) -> usize {
            self.alignment
        }

        #[inline(always)]
        pub const fn len(&self) -> usize {
            self.len
        }

        /// Cardinality may be queried safely without exposing uninitialized data.
        #[inline(always)]
        pub const fn numel(&self) -> usize { self.shape.numel() }

        #[inline(always)]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// # Safety
        /// Every element in the allocation must have been fully initialized with a
        /// valid `T` before this function is called.
        pub unsafe fn assume_init(mut self) -> Tensor<T, R> {
            let data = AlignedBuffer {
                ptr: self.ptr,
                len: self.len,
                alignment: self.alignment,
                layout: self.layout.take(),
                vec_capacity: None,
                _marker: PhantomData,
            };
            self.ptr = NonNull::dangling();
            self.len = 0;
            Tensor {
                shape: self.shape,
                data,
            }
        }
    }

    impl<T: TensorElement, const R: usize> Drop for UninitTensor<T, R>
    where
        Rank<R>: ValidRank,
    {
        fn drop(&mut self) {
            if let Some(layout) = self.layout {
                unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
            }
        }
    }
}
pub use owned::{Tensor, UninitTensor};



// =============================================================================
// Host-side tensor construction
// =============================================================================

/// Reusable, validated allocation metadata for repeated creation of one shape.
///
/// This is deliberately small and immutable. It is a tensor-allocation helper,
/// not an execution plan and not a runtime performance-selection mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CreationPlan<const R: usize>
where
    Rank<R>: ValidRank,
{
    shape: Shape<R>,
    alignment: usize,
}

impl<const R: usize> CreationPlan<R>
where
    Rank<R>: ValidRank,
{
    #[inline]
    pub(crate) fn new(dims: [usize; R]) -> Result<Self, TensorError> {
        Self::with_alignment(dims, DEFAULT_HOST_ALIGNMENT)
    }

    pub fn with_alignment(
        dims: [usize; R],
        alignment: usize,
    ) -> Result<Self, TensorError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(TensorError::InvalidAlignment);
        }
        Ok(Self {
            shape: Shape::contiguous(dims)?,
            alignment,
        })
    }

    #[inline(always)]
    pub const fn shape(&self) -> &Shape<R> {
        &self.shape
    }

    #[inline(always)]
    pub const fn dims(&self) -> &[usize; R] {
        self.shape.dims()
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.shape.len()
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    #[inline(always)]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }
}

impl<T: TensorElement, const R: usize> Tensor<T, R>
where
    Rank<R>: ValidRank,
{
    /// Zero-filled contiguous tensor using the default host alignment.
    #[inline]
    pub fn zeros(dims: [usize; R]) -> Result<Self, TensorError> {
        Self::zeros_aligned(dims, DEFAULT_HOST_ALIGNMENT)
    }

    pub fn zeros_aligned(
        dims: [usize; R],
        alignment: usize,
    ) -> Result<Self, TensorError> {
        let plan = CreationPlan::with_alignment(dims, alignment)?;
        Self::zeros_with_plan(plan)
    }

    /// Creates a zero-filled tensor from already validated creation metadata.
    pub(crate) fn zeros_with_plan(plan: CreationPlan<R>) -> Result<Self, TensorError> {
        let data = AlignedBuffer::zeroed_aligned(plan.len(), plan.alignment())?;
        Ok(Self {
            shape: *plan.shape(),
            data,
        })
    }

    /// Constructs a tensor whose logical elements are all `value`.
    #[inline]
    pub fn new(dims: [usize; R], value: T) -> Result<Self, TensorError> {
        Self::full(dims, value)
    }

    #[inline]
    pub fn full(dims: [usize; R], value: T) -> Result<Self, TensorError> {
        Self::full_aligned(dims, value, DEFAULT_HOST_ALIGNMENT)
    }

    /// Compatibility alias for code that prefers the `filled` spelling.
    #[inline]
    pub fn filled(dims: [usize; R], value: T) -> Result<Self, TensorError> {
        Self::full(dims, value)
    }

    pub fn full_aligned(
        dims: [usize; R],
        value: T,
        alignment: usize,
    ) -> Result<Self, TensorError> {
        let plan = CreationPlan::with_alignment(dims, alignment)?;
        Self::full_with_plan(plan, value)
    }

    /// Compatibility alias for earlier releases.
    #[inline]
    pub fn filled_aligned(
        dims: [usize; R],
        value: T,
        alignment: usize,
    ) -> Result<Self, TensorError> {
        Self::full_aligned(dims, value, alignment)
    }

    /// Creates a full tensor without a redundant zero-initialization pass.
    pub(crate) fn full_with_plan(
        plan: CreationPlan<R>,
        value: T,
    ) -> Result<Self, TensorError> {
        let mut out = UninitTensor::<T, R>::from_plan(plan)?;
        let len = out.len();
        let ptr = out.as_mut_ptr();
        let mut i = 0usize;
        while i < len {
            unsafe { ptr.add(i).write(value) };
            i += 1;
        }
        Ok(unsafe { out.assume_init() })
    }

    /// Allocates output storage without initializing it.
    ///
    /// The returned type cannot be read as a tensor until the caller has fully
    /// initialized every element and explicitly calls `assume_init`.
    #[inline]
    pub fn empty_uninit(dims: [usize; R]) -> Result<UninitTensor<T, R>, TensorError> {
        UninitTensor::new(dims)
    }

    #[inline]
    pub(crate) fn empty_uninit_with_plan(
        plan: CreationPlan<R>,
    ) -> Result<UninitTensor<T, R>, TensorError> {
        UninitTensor::from_plan(plan)
    }

    #[inline]
    pub fn zeros_like(&self) -> Result<Self, TensorError> {
        Self::zeros_aligned(*self.shape.dims(), self.alignment())
    }

    #[inline]
    pub fn full_like(&self, value: T) -> Result<Self, TensorError> {
        Self::full_aligned(*self.shape.dims(), value, self.alignment())
    }

    #[inline]
    pub fn empty_like_uninit(&self) -> Result<UninitTensor<T, R>, TensorError> {
        UninitTensor::new_aligned(*self.shape.dims(), self.alignment())
    }

    pub fn from_slice(dims: [usize; R], src: &[T]) -> Result<Self, TensorError> {
        Self::from_slice_aligned(dims, src, DEFAULT_HOST_ALIGNMENT)
    }

    pub fn from_slice_aligned(
        dims: [usize; R],
        src: &[T],
        alignment: usize,
    ) -> Result<Self, TensorError> {
        let shape = Shape::contiguous(dims)?;
        if shape.len() != src.len() {
            return Err(TensorError::LengthMismatch);
        }
        let data = AlignedBuffer::from_slice_aligned(src, alignment)?;
        Ok(Self { shape, data })
    }

    /// Internal linear constructor for operations that do not need N-D
    /// coordinates. Avoiding the odometer used by `from_fn` removes rank-sized
    /// index maintenance from dense elementwise creation hot paths.
    pub(crate) fn from_linear_fn_aligned<F>(
        dims: [usize; R],
        alignment: usize,
        f: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut(usize) -> T,
    {
        let plan = CreationPlan::with_alignment(dims, alignment)?;
        Self::from_linear_fn_with_plan(plan, f)
    }

    /// A validated, budget-checked plan avoids a redundant shape traversal.
    pub(crate) fn from_linear_fn_with_plan<F>(
        plan: CreationPlan<R>,
        mut f: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut(usize) -> T,
    {
        let mut out = UninitTensor::<T, R>::from_plan(plan)?;
        let len = out.len();
        let ptr = out.as_mut_ptr();
        let mut linear = 0usize;
        while linear < len {
            unsafe { ptr.add(linear).write(f(linear)) };
            linear += 1;
        }
        Ok(unsafe { out.assume_init() })
    }

    /// Public host initialization helper over compile-time-rank indices.
    /// This writes a newly allocated buffer, not a runtime graph kernel;
    /// numerical graph execution remains confined to UASDEA replay.
    pub fn from_fn<F>(dims: [usize; R], f: F) -> Result<Self, TensorError>
    where
        F: FnMut([usize; R]) -> T,
    {
        Self::from_fn_aligned(dims, DEFAULT_HOST_ALIGNMENT, f)
    }

    pub fn from_fn_aligned<F>(
        dims: [usize; R],
        alignment: usize,
        f: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut([usize; R]) -> T,
    {
        let plan = CreationPlan::with_alignment(dims, alignment)?;
        Self::from_fn_with_plan(plan, f)
    }

    /// Build from checked metadata once; callback receives a stack index array.
    pub(crate) fn from_fn_with_plan<F>(
        plan: CreationPlan<R>,
        mut f: F,
    ) -> Result<Self, TensorError>
    where
        F: FnMut([usize; R]) -> T,
    {
        let dims = *plan.dims();
        let mut out = UninitTensor::<T, R>::from_plan(plan)?;
        let len = out.len();
        let ptr = out.as_mut_ptr();
        let mut index = [0usize; R];
        let mut linear = 0usize;

        while linear < len {
            unsafe { ptr.add(linear).write(f(index)) };
            linear += 1;
            if linear == len {
                break;
            }

            let mut axis = R;
            while axis > 0 {
                axis -= 1;
                index[axis] += 1;
                if index[axis] < dims[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }

        Ok(unsafe { out.assume_init() })
    }
}

impl<T: TensorElement> Tensor<T, 0> {
    #[inline]
    pub fn scalar(value: T) -> Result<Self, TensorError> {
        Self::full([], value)
    }
}

impl<const R: usize> Tensor<f32, R>
where
    Rank<R>: ValidRank,
{
    #[inline]
    pub fn ones(dims: [usize; R]) -> Result<Self, TensorError> {
        Self::full(dims, 1.0)
    }

    #[inline]
    pub fn ones_like(&self) -> Result<Self, TensorError> {
        Self::full_aligned(*self.shape.dims(), 1.0, self.alignment())
    }

    /// Deterministic uniform random values in `[0, 1)` using SplitMix64.
    pub fn randu(dims: [usize; R], seed: u64) -> Result<Self, TensorError> {
        let mut rng = SplitMix64::new(seed);
        Self::from_linear_fn_aligned(dims, DEFAULT_HOST_ALIGNMENT, |_| rng.next_unit_f32())
    }

    #[inline]
    pub fn randu_like(&self, seed: u64) -> Result<Self, TensorError> {
        let mut rng = SplitMix64::new(seed);
        Self::from_linear_fn_aligned(*self.shape.dims(), self.alignment(), |_| rng.next_unit_f32())
    }

    /// Seed-reproducible standard-normal values via the Box-Muller transform.
    /// Floating-point transcendental precision is target/toolchain dependent:
    /// do not treat these host initializers as bitwise cross-platform kernels.
    pub fn randn(dims: [usize; R], seed: u64) -> Result<Self, TensorError> {
        let mut rng = SplitMix64::new(seed);
        let mut spare: Option<f32> = None;
        Self::from_linear_fn_aligned(dims, DEFAULT_HOST_ALIGNMENT, |_| {
            if let Some(value) = spare.take() {
                return value;
            }
            // The logarithm's input must be positive. Clamping a sampled zero
            // to f32::MIN_POSITIVE introduced a spurious ~13-sigma tail value.
            // The 24-bit uniform grid is mapped to (0, 1], so its smallest
            // value is 2^-24, with no extra RNG call, rejection, or branch.
            let u1 = rng.next_positive_unit_f32();
            let u2 = rng.next_unit_f32();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = core::f32::consts::TAU * u2;
            let (sin, cos) = angle.sin_cos();
            spare = Some(radius * sin);
            radius * cos
        })
    }

    pub fn randn_like(&self, seed: u64) -> Result<Self, TensorError> {
        let mut rng = SplitMix64::new(seed);
        let mut spare: Option<f32> = None;
        Self::from_linear_fn_aligned(*self.shape.dims(), self.alignment(), |_| {
            if let Some(value) = spare.take() {
                return value;
            }
            // The logarithm's input must be positive. Clamping a sampled zero
            // to f32::MIN_POSITIVE introduced a spurious ~13-sigma tail value.
            // The 24-bit uniform grid is mapped to (0, 1], so its smallest
            // value is 2^-24, with no extra RNG call, rejection, or branch.
            let u1 = rng.next_positive_unit_f32();
            let u2 = rng.next_unit_f32();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = core::f32::consts::TAU * u2;
            let (sin, cos) = angle.sin_cos();
            spare = Some(radius * sin);
            radius * cos
        })
    }

    /// Integer-valued F32 tensor sampled uniformly from `[low, high)`.
    pub fn randint(
        dims: [usize; R],
        low: i32,
        high: i32,
        seed: u64,
    ) -> Result<Self, TensorError> {
        let width = exact_f32_integer_width(low, high)?;
        let mut rng = SplitMix64::new(seed);
        let threshold = SplitMix64::rejection_threshold(width);
        Self::from_linear_fn_aligned(dims, DEFAULT_HOST_ALIGNMENT, |_| {
            let offset = rng.next_bounded_with_threshold(width, threshold) as i64;
            (i64::from(low) + offset) as f32
        })
    }

    pub fn randint_like(
        &self,
        low: i32,
        high: i32,
        seed: u64,
    ) -> Result<Self, TensorError> {
        let width = exact_f32_integer_width(low, high)?;
        let mut rng = SplitMix64::new(seed);
        let threshold = SplitMix64::rejection_threshold(width);
        Self::from_linear_fn_aligned(*self.shape.dims(), self.alignment(), |_| {
            let offset = rng.next_bounded_with_threshold(width, threshold) as i64;
            (i64::from(low) + offset) as f32
        })
    }

    /// F32 mask with values exactly `0.0` or `1.0`.
    pub fn bernoulli(
        dims: [usize; R],
        probability: f32,
        seed: u64,
    ) -> Result<Self, TensorError> {
        if !(0.0..=1.0).contains(&probability) || probability.is_nan() {
            return Err(TensorError::InvalidProbability);
        }
        if probability == 0.0 { return Self::zeros(dims); }
        if probability == 1.0 { return Self::ones(dims); }
        // The acceptance threshold is derived once from the exact f32 value.
        // Compare full 64-bit RNG words instead of discarding 40 bits and
        // converting each sample to f32. This improves tail probabilities and
        // eliminates per-sample floating-point conversion/multiplication.
        let threshold = SplitMix64::bernoulli_threshold(probability);
        let mut rng = SplitMix64::new(seed);
        Self::from_linear_fn_aligned(dims, DEFAULT_HOST_ALIGNMENT, |_| {
            if rng.next_u64() < threshold { 1.0 } else { 0.0 }
        })
    }

    pub fn bernoulli_like(
        &self,
        probability: f32,
        seed: u64,
    ) -> Result<Self, TensorError> {
        if !(0.0..=1.0).contains(&probability) || probability.is_nan() {
            return Err(TensorError::InvalidProbability);
        }
        if probability == 0.0 { return self.zeros_like(); }
        if probability == 1.0 { return self.full_like(1.0); }
        let threshold = SplitMix64::bernoulli_threshold(probability);
        let mut rng = SplitMix64::new(seed);
        Self::from_linear_fn_aligned(*self.shape.dims(), self.alignment(), |_| {
            if rng.next_u64() < threshold { 1.0 } else { 0.0 }
        })
    }

    /// Fills this tensor with a flattened linspace while preserving its shape.
    pub fn linspace_like(&self, start: f32, end: f32) -> Result<Self, TensorError> {
        let len = self.len();
        if len == 0 {
            return Self::from_slice_aligned(*self.shape.dims(), &[], self.alignment());
        }
        if len == 1 || start.to_bits() == end.to_bits() {
            // Bit-identical endpoints (including infinities or NaNs) produce
            // a bit-identical constant sequence, with no interpolation work.
            return Self::full_aligned(*self.shape.dims(), start, self.alignment());
        }
        // Calculate the step once in f64. An f32 subtraction of opposite
        // large finite endpoints could overflow to infinity even though every
        // interpolated result is representable in f32. Using f64 also avoids
        // loss of distinct positions when the element count exceeds 2^24.
        let delta = f64::from(end) - f64::from(start);
        let inverse_intervals = 1.0f64 / (len - 1) as f64;
        Self::from_linear_fn_aligned(*self.shape.dims(), self.alignment(), |linear| {
            stable_linspace_value(start, end, linear, len, delta, inverse_intervals)
        })
    }
}

/// Host-only f32 tensor initialization (not a runtime numeric kernel).
/// Hoist the interval arithmetic out of the per-element loop and preserve
/// both requested endpoints bit-for-bit, including signed zero and NaN payloads.
#[inline(always)]
fn stable_linspace_value(
    start: f32,
    end: f32,
    index: usize,
    count: usize,
    delta: f64,
    inverse_intervals: f64,
) -> f32 {
    if index == 0 { return start; }
    if index == count - 1 { return end; }
    // For finite f32 endpoints, their difference and this interpolation are
    // finite in f64. Conversion to f32 rounds only the final stored result.
    (f64::from(start) + delta * ((index as f64) * inverse_intervals)) as f32
}

impl Tensor<f32, 0> {
    #[inline]
    pub fn scalar_zero() -> Result<Self, TensorError> {
        Self::scalar(0.0)
    }

    #[inline]
    pub fn scalar_one() -> Result<Self, TensorError> {
        Self::scalar(1.0)
    }
}

impl Tensor<f32, 1> {
    /// Integer arange stored as F32 values. `step` must be nonzero.
    pub fn arange(start: i64, end: i64, step: i64) -> Result<Self, TensorError> {
        if step == 0 {
            return Err(TensorError::InvalidRange);
        }
        let len = arange_len(start, end, step)?;
        // Common dense index vector: the linear constructor already visits
        // 0..len in increasing order. Avoid maintaining a second running
        // integer and enable a simple straight-line conversion loop.
        if start == 0 && step == 1 {
            return Self::from_linear_fn_aligned(
                [len], DEFAULT_HOST_ALIGNMENT, |index| index as f32,
            );
        }
        let mut current = start;
        Self::from_linear_fn_aligned([len], DEFAULT_HOST_ALIGNMENT, |_| {
            let value = current as f32;
            // The checked arange_len guarantees that EVERY element that is
            // subsequently observed stays within i64. The only update that
            // could overflow happens after the last value was written, and
            // that update is never read. Wrapping arithmetic thus gives
            // bit-for-bit identical output without saturation work per item.
            current = current.wrapping_add(step);
            value
        })
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Self, TensorError> {
        if steps == 0 {
            return Self::from_slice([0], &[]);
        }
        if steps == 1 || start.to_bits() == end.to_bits() {
            return Self::full([steps], start);
        }
        let delta = f64::from(end) - f64::from(start);
        let inverse_intervals = 1.0f64 / (steps - 1) as f64;
        Self::from_linear_fn_aligned([steps], DEFAULT_HOST_ALIGNMENT, |linear| {
            stable_linspace_value(start, end, linear, steps, delta, inverse_intervals)
        })
    }

    /// Deterministic permutation of `0..n`, represented as F32 values.
    ///
    /// F32 represents every integer exactly through 2^24 (inclusive). The
    /// exclusive permutation length may therefore be 2^24+1: the largest
    /// generated value is only 2^24. Larger lengths are rejected rather than
    /// silently generating duplicate f32 integer values.
    pub fn randperm(n: usize, seed: u64) -> Result<Self, TensorError> {
        if !randperm_length_is_exact_f32(n) {
            return Err(TensorError::InvalidRange);
        }
        let mut out = Self::from_linear_fn_aligned(
            [n],
            DEFAULT_HOST_ALIGNMENT,
            |linear| linear as f32,
        )?;
        let mut rng = SplitMix64::new(seed);
        let data = out.as_mut_slice();
        let mut i = n;
        while i > 1 {
            i -= 1;
            let j = rng.next_bounded((i + 1) as u64) as usize;
            data.swap(i, j);
        }
        Ok(out)
    }
}

impl Tensor<f32, 2> {
    pub fn eye(rows: usize, cols: usize) -> Result<Self, TensorError> {
        Self::eye_aligned(rows, cols, DEFAULT_HOST_ALIGNMENT)
    }

    fn eye_aligned(rows: usize, cols: usize, alignment: usize) -> Result<Self, TensorError> {
        let mut out = Self::zeros_aligned([rows, cols], alignment)?;
        let diagonal = rows.min(cols);
        if diagonal == 0 {
            return Ok(out);
        }
        let step = cols.checked_add(1).ok_or(TensorError::ShapeOverflow)?;
        let data = out.as_mut_slice();
        let mut i = 0usize;
        let mut offset = 0usize;
        while i < diagonal {
            data[offset] = 1.0;
            offset += step;
            i += 1;
        }
        Ok(out)
    }

    #[inline]
    pub fn eye_square(size: usize) -> Result<Self, TensorError> {
        Self::eye(size, size)
    }

    pub fn eye_like(&self) -> Result<Self, TensorError> {
        let dims = *self.shape.dims();
        Self::eye_aligned(dims[0], dims[1], self.alignment())
    }
}

impl<T: TensorElement, const R: usize> UninitTensor<T, R>
where
    Rank<R>: ValidRank,
{
    #[inline]
    pub(crate) fn from_plan(plan: CreationPlan<R>) -> Result<Self, TensorError> {
        // CreationPlan is private to the crate and constructed only by
        // checked constructors; reuse its exact Shape rather than validating
        // and recomputing every stride/extent a second time.
        Self::allocate_validated_shape(*plan.shape(), plan.alignment())
    }
}

/// An f32 cannot exactly represent every integer outside [-2^24, 2^24].
/// Keep the integer sampler's advertised discrete, uniform output contract;
/// `high` is exclusive, so 2^24 + 1 is a valid upper boundary.
#[inline]
fn exact_f32_integer_width(low: i32, high: i32) -> Result<u64, TensorError> {
    const MAX_EXACT_INT: i64 = 1 << 24;
    let low = i64::from(low);
    let high = i64::from(high);
    if high <= low || low < -MAX_EXACT_INT || high > MAX_EXACT_INT + 1 {
        return Err(TensorError::InvalidRange);
    }
    // The range is positive, nonempty and at most 2 * 2^24 + 1.
    Ok((high - low) as u64)
}

/// A permutation of n elements produces the integers 0..n-1. The maximum
/// exactly representable consecutive f32 integer is 2^24, so n can be one
/// greater. This check is constant time and performs no allocation.
#[inline(always)]
const fn randperm_length_is_exact_f32(n: usize) -> bool {
    n <= 16_777_217
}

#[inline]
fn arange_len(start: i64, end: i64, step: i64) -> Result<usize, TensorError> {
    if step == 0 {
        return Err(TensorError::InvalidRange);
    }
    let count = if step > 0 {
        if start >= end {
            0i128
        } else {
            let distance = i128::from(end) - i128::from(start);
            (distance + i128::from(step) - 1) / i128::from(step)
        }
    } else if start <= end {
        0i128
    } else {
        let step_abs = -i128::from(step);
        let distance = i128::from(start) - i128::from(end);
        (distance + step_abs - 1) / step_abs
    };
    usize::try_from(count).map_err(|_| TensorError::ShapeOverflow)
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline(always)]
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    #[inline(always)]
    fn next_unit_f32(&mut self) -> f32 {
        // Use the high 24 bits, matching the effective precision of an F32 mantissa.
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform on the 24-bit f32 grid (0, 1], for log-based transforms.
    /// Subtraction is exact for this grid, and avoids artificial tiny values.
    #[inline(always)]
    fn next_positive_unit_f32(&mut self) -> f32 {
        1.0 - self.next_unit_f32()
    }

    /// Number of accepted raw RNG words for a Bernoulli probability on the
    /// full 2^64-point grid. Called only once per output tensor. f32 -> f64
    /// conversion is exact; for p in (0, 1), the product is below 2^64 and
    /// conversion to u64 truncates only genuinely sub-grid probabilities.
    #[inline(always)]
    fn bernoulli_threshold(probability: f32) -> u64 {
        // The caller handles the exact p=0 and p=1 endpoints before sampling.
        debug_assert!(probability > 0.0 && probability < 1.0);
        const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
        (f64::from(probability) * TWO_POW_64) as u64
    }

    #[inline(always)]
    fn rejection_threshold(bound: u64) -> u64 {
        debug_assert!(bound > 0);
        bound.wrapping_neg() % bound
    }

    #[inline(always)]
    fn next_bounded(&mut self, bound: u64) -> u64 {
        self.next_bounded_with_threshold(bound, Self::rejection_threshold(bound))
    }

    #[inline(always)]
    fn next_bounded_with_threshold(&mut self, bound: u64, threshold: u64) -> u64 {
        debug_assert!(bound > 0);
        // Lemire-style unbiased multiply-high + rejection sampling.
        loop {
            let product = u128::from(self.next_u64()) * u128::from(bound);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as u64;
            }
        }
    }
}

pub use view::{SliceSpec, TensorView, TensorViewMut};
pub use signed_view::{SignedSliceSpec, SignedTensorView};
pub use signed_view_mut::SignedTensorViewMut;
pub use axis::ReductionAxes;
pub use numerics::{canonical_maximum, canonical_minimum, replace_argmax, replace_argmin,
                   CANONICAL_QUIET_NAN32};

#[inline(always)]
pub(crate) fn normalize_axis(axis: isize, rank: usize) -> Result<usize, TensorError> {
    if axis >= 0 {
        let normalized = axis as usize;
        if normalized < rank {
            return Ok(normalized);
        }
        return Err(TensorError::AxisOutOfBounds);
    }

    let magnitude = axis.unsigned_abs();
    if magnitude == 0 || magnitude > rank {
        return Err(TensorError::AxisOutOfBounds);
    }
    Ok(rank - magnitude)
}

#[inline(always)]
pub(crate) fn normalize_index(index: isize, dim: usize) -> Result<usize, TensorError> {
    if index >= 0 {
        let normalized = index as usize;
        if normalized < dim {
            return Ok(normalized);
        }
        return Err(TensorError::IndexOutOfBounds);
    }

    let magnitude = index.unsigned_abs();
    if magnitude == 0 || magnitude > dim {
        return Err(TensorError::IndexOutOfBounds);
    }
    Ok(dim - magnitude)
}

#[cfg(test)]
#[path = "../../tests/tensor_light/unit/tensor/checked_host_allocation_tests.rs"]
mod checked_host_allocation_tests;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/tensor/precomputed_bounded_rng_tests.rs"]
mod precomputed_bounded_rng_tests;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/tensor/randperm_length_boundary_tests.rs"]
mod randperm_length_boundary_tests;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/tensor/bernoulli_full_precision_tests.rs"]
mod bernoulli_full_precision_tests;
