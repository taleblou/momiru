// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | EXECUTION ENGINE
// ==============================================================================
//
// Project:       Momiru
// Component:     Execution
// Subcomponent:  IR
// Module:        momiru::tensor_light::execution::ir
// Source:        src/tensor_light/execution/ir.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Canonical semantic intermediate representation for UASDEA static dataflow
//   and exact workload identity.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Represent tensor and operation semantics in canonical graph records.
//   - Define explicit workload and shape contracts.
//   - Keep resource placement, target artifacts and schedules outside semantic
//     IR.
//   - Produce stable identities for semantically equivalent workload definitions.
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

//! Canonical UASDEA intermediate representation.
//!
//! This module defines tensor semantics, operation descriptors, the canonical
//! static dataflow graph, workload contracts, and exact workload identity.
//! Placement, resources, schedules, artifacts, and shared validation errors are not
//! part of this semantic IR.

use core::hash::{Hash, Hasher};

use crate::tensor_light::tensor::{DType, MAX_RANK};
use super::target::{gpu_feature, HardwareContract};
use super::{stable_hash_pair, PlanError, StableIdentityHasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub signature_hi: u64,
    pub signature_lo: u64,
}




// =============================================================================
// Canonical static tensor/dataflow descriptors
// =============================================================================

pub type TensorId = u32;
pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StaticShape {
    Rank0,
    Rank1([usize; 1]),
    Rank2([usize; 2]),
    Rank3([usize; 3]),
    Rank4([usize; 4]),
    Rank5([usize; 5]),
    Rank6([usize; 6]),
    Rank7([usize; 7]),
    Rank8([usize; 8]),
}

impl StaticShape {
    #[inline(always)]
    pub const fn rank(&self) -> usize {
        match self {
            Self::Rank0 => 0,
            Self::Rank1(_) => 1,
            Self::Rank2(_) => 2,
            Self::Rank3(_) => 3,
            Self::Rank4(_) => 4,
            Self::Rank5(_) => 5,
            Self::Rank6(_) => 6,
            Self::Rank7(_) => 7,
            Self::Rank8(_) => 8,
        }
    }

    #[inline(always)]
    pub const fn dim(&self, axis: usize) -> Option<usize> {
        match self {
            Self::Rank0 => None,
            Self::Rank1(d) => {
                if axis < 1 { Some(d[axis]) } else { None }
            }
            Self::Rank2(d) => {
                if axis < 2 { Some(d[axis]) } else { None }
            }
            Self::Rank3(d) => {
                if axis < 3 { Some(d[axis]) } else { None }
            }
            Self::Rank4(d) => {
                if axis < 4 { Some(d[axis]) } else { None }
            }
            Self::Rank5(d) => {
                if axis < 5 { Some(d[axis]) } else { None }
            }
            Self::Rank6(d) => {
                if axis < 6 { Some(d[axis]) } else { None }
            }
            Self::Rank7(d) => {
                if axis < 7 { Some(d[axis]) } else { None }
            }
            Self::Rank8(d) => {
                if axis < 8 { Some(d[axis]) } else { None }
            }
        }
    }

    pub fn element_count(&self) -> Option<usize> {
        // Empty shapes have zero elements regardless of oversized nonzero axes.
        for axis in 0..self.rank() {
            if self.dim(axis)? == 0 { return Some(0); }
        }
        let mut count = 1usize;
        let mut axis = 0usize;
        while axis < self.rank() {
            count = count.checked_mul(self.dim(axis)?)?;
            axis += 1;
        }
        Some(count)
    }

    /// Checked host-addressable byte count for the v24 fixed-rank value ABI.
    /// A multiplication that fits `usize` can still exceed Rust's maximum
    /// addressable slice/object byte length (`isize::MAX`). The native device
    /// allocator has its own independently checked width/resource contracts.
    /// Empty logical shapes are zero bytes before multiplying large extents.
    pub fn byte_count(&self, dtype: DType) -> Option<usize> {
        let bytes = dtype.packed_byte_count(self.element_count()?)?;
        if bytes > isize::MAX as usize { return None; }
        Some(bytes)
    }
}

/// NumPy-style exact pairwise broadcast extent. In particular, 0 versus 1
/// broadcasts to ZERO (not max(0, 1) = 1). This shared semantic primitive is
/// used by elementwise/select and batched matmul shape verification, never by
/// runtime scheduling, allocation or kernel selection.
#[inline(always)]
pub(crate) const fn canonical_broadcast_extent(lhs: usize, rhs: usize) -> Option<usize> {
    if lhs == rhs { Some(lhs) }
    else if lhs == 1 { Some(rhs) }
    else if rhs == 1 { Some(lhs) }
    else { None }
}

/// Canonical compact descriptor for standard trailing-axis broadcasting.
/// Numeric/mask elementwise operations use one or two inputs; `Select` uses
/// three. The legacy one/two-input bit layout is preserved exactly so adding a
/// ternary input cannot perturb existing workload identities. Input 2 uses
/// bits 32..39 for per-axis broadcast flags and bits 40..43 for its rank.
/// Exact input/output shapes remain in the canonical graph; this field makes
/// the broadcast schedule contract deterministic rather than caller-defined.
pub fn canonical_elementwise_broadcast_signature(
    input_shapes: &[StaticShape],
    output: StaticShape,
) -> Result<u64, PlanError> {
    if input_shapes.is_empty() || input_shapes.len() > 3 {
        return Err(PlanError::InvalidOperationSpec);
    }
    let output_rank = output.rank();
    // Elementwise and Select produce the true right-aligned broadcast of
    // their inputs. A separately named Broadcast/Expand view can enlarge a
    // singleton; elementwise arithmetic itself must NEVER manufacture an
    // arbitrary larger result, a fake leading axis or a nonempty 0-vs-1 axis.
    let expected_rank = input_shapes.iter().map(StaticShape::rank).max()
        .ok_or(PlanError::InvalidOperationSpec)?;
    if output_rank != expected_rank {
        return Err(PlanError::InvalidOperationSpec);
    }
    for axis in 0..output_rank {
        let mut required = 1usize;
        for input in input_shapes {
            let source = if axis + input.rank() < output_rank {
                1usize
            } else {
                input.dim(axis + input.rank() - output_rank)
                    .ok_or(PlanError::InvalidOperationSpec)?
            };
            required = canonical_broadcast_extent(required, source)
                .ok_or(PlanError::InvalidOperationSpec)?;
        }
        if output.dim(axis) != Some(required) {
            return Err(PlanError::InvalidOperationSpec);
        }
    }
    let mut signature = (output_rank as u64) << 24;
    let mut any_broadcast = false;
    let mut input_index = 0usize;
    while input_index < input_shapes.len() {
        let input = input_shapes[input_index];
        let input_rank = input.rank();
        if input_rank > output_rank {
            return Err(PlanError::InvalidOperationSpec);
        }
        let rank_shift = if input_index < 2 {
            16 + input_index * 4
        } else {
            40
        };
        let broadcast_shift = if input_index < 2 {
            input_index * MAX_RANK
        } else {
            32
        };
        signature |= (input_rank as u64) << rank_shift;
        let leading = output_rank - input_rank;
        let mut output_axis = 0usize;
        while output_axis < output_rank {
            let output_dim = output.dim(output_axis).ok_or(PlanError::InvalidOperationSpec)?;
            let broadcast = if output_axis < leading {
                true
            } else {
                let input_dim = input
                    .dim(output_axis - leading)
                    .ok_or(PlanError::InvalidOperationSpec)?;
                if input_dim != 1 && input_dim != output_dim {
                    return Err(PlanError::InvalidOperationSpec);
                }
                input_dim == 1 && output_dim != 1
            };
            if broadcast {
                signature |= 1u64 << (broadcast_shift + output_axis);
                any_broadcast = true;
            }
            output_axis += 1;
        }
        if input != output {
            any_broadcast = true;
        }
        input_index += 1;
    }
    if any_broadcast {
        Ok(signature | (1u64 << 63))
    } else {
        Ok(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LayoutKind {
    RowMajorContiguous,
    ColumnMajorContiguous,
    Strided,
    Blocked,
    Packed,
    ChannelsLast,
    TargetNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    pub kind: LayoutKind,
    pub strides: [isize; MAX_RANK],
    pub block: [u16; MAX_RANK],
    pub alignment_bytes: u32,
}

impl TensorLayout {
    /// Verifies canonical invariants that can be proven from shape + layout alone.
    /// Target-native/blocked/packed layouts keep their offline-frozen descriptors.
    /// A contiguous layout must have exact strides only on non-singleton axes:
    /// singleton axes have zero index displacement, hence arbitrary strides.
    /// The stored strides are nonetheless exact inputs to the frozen guard.
    pub fn validate_for(&self, shape: &StaticShape) -> Result<(), PlanError> {
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(PlanError::InvalidTensorAlignment);
        }

        let rank = shape.rank();
        let mut axis = rank;
        while axis < MAX_RANK {
            if self.strides[axis] != 0 || self.block[axis] != 0 {
                return Err(PlanError::InvalidTensorLayoutRank);
            }
            axis += 1;
        }

        // Non-blocked logical layouts may not smuggle blocking metadata into
        // otherwise canonical stride descriptors. Packed/blocked/target-native
        // layouts own their block descriptors explicitly.
        if matches!(
            self.kind,
            LayoutKind::RowMajorContiguous
                | LayoutKind::ColumnMajorContiguous
                | LayoutKind::Strided
                | LayoutKind::ChannelsLast
        ) {
            let mut i = 0usize;
            while i < rank {
                if self.block[i] != 0 {
                    return Err(PlanError::InvalidTensorLayout);
                }
                i += 1;
            }
        }

        // Empty C/F mappings are vacuously contiguous and have no physical
        // elements; do not reject solely due to stride-product overflow.
        if shape.element_count() == Some(0) && matches!(self.kind,
            LayoutKind::RowMajorContiguous | LayoutKind::ColumnMajorContiguous) {
            return Ok(());
        }
        match self.kind {
            LayoutKind::RowMajorContiguous => {
                let mut expected = 1isize;
                let mut i = rank;
                while i > 0 {
                    i -= 1;
                    let dim = isize::try_from(shape.dim(i).ok_or(PlanError::InvalidTensorLayout)?)
                        .map_err(|_| PlanError::InvalidTensorLayout)?;
                    // A singleton axis contributes zero to every physical
                    // offset, so its stride is semantically irrelevant. This
                    // matches Shape::is_c_contiguous and the frozen f32 layout
                    // semantics; the actual stride remains in the plan guard.
                    if dim > 1 && self.strides[i] != expected {
                        return Err(PlanError::InvalidTensorLayout);
                    }
                    expected = expected
                        .checked_mul(dim)
                        .ok_or(PlanError::InvalidTensorLayout)?;
                }
            }
            LayoutKind::ColumnMajorContiguous => {
                let mut expected = 1isize;
                let mut i = 0usize;
                while i < rank {
                    let dim = isize::try_from(shape.dim(i).ok_or(PlanError::InvalidTensorLayout)?)
                        .map_err(|_| PlanError::InvalidTensorLayout)?;
                    // As with C order, a dimension of extent one has no
                    // observable stride. Keep its exact stride in the signed
                    // descriptor/key, but do not reject a valid F-contiguous
                    // logical mapping solely because it is noncanonical.
                    if dim > 1 && self.strides[i] != expected {
                        return Err(PlanError::InvalidTensorLayout);
                    }
                    expected = expected
                        .checked_mul(dim)
                        .ok_or(PlanError::InvalidTensorLayout)?;
                    i += 1;
                }
            }
            LayoutKind::ChannelsLast => {
                // NCHW and NCDHW keep their logical axes in place while
                // placing channels innermost in physical element order.
                // Singleton axes have only coordinate zero, so their stride
                // cannot affect the physical sequence. This accepts the
                // legitimate C/channels-last ambiguity (e.g. N1HW) while
                // preserving an EXACT stride guard in the frozen plan.
                // Empty mappings also have no observable physical ordering.
                let physical_fast_to_slow: &[usize] = match rank {
                    4 => &[1, 3, 2, 0],
                    5 => &[1, 4, 3, 2, 0],
                    _ => return Err(PlanError::InvalidTensorLayout),
                };
                if shape.element_count() == Some(0) {
                    return Ok(());
                }
                let mut expected = 1isize;
                for &axis in physical_fast_to_slow {
                    let dim = isize::try_from(shape.dim(axis).ok_or(PlanError::InvalidTensorLayout)?)
                        .map_err(|_| PlanError::InvalidTensorLayout)?;
                    if dim > 1 && self.strides[axis] != expected {
                        return Err(PlanError::InvalidTensorLayout);
                    }
                    expected = expected.checked_mul(dim)
                        .ok_or(PlanError::InvalidTensorLayout)?;
                }
            }
            LayoutKind::Strided => {
                // Signed and zero strides are canonical. A frozen physical storage
                // carries an explicit byte base offset, so reversed, narrowed and
                // broadcast views can be represented exactly without materializing.
                // Writable overlap and address-span safety are validated against the
                // concrete TensorStoragePlan after offline placement.
            }
            LayoutKind::Blocked | LayoutKind::Packed | LayoutKind::TargetNative => {}
        }

        Ok(())
    }

    /// Dtype-aware physical alignment gate for DIRECT primitive values.
    /// Structural stride/layout validity alone is not proof that an f32,
    /// integer index, or u64 result can be accessed with an aligned typed
    /// load. The ABI reserves the natural-width alignment on every supported
    /// target (a conservative cross-target floor for 64-bit integers).
    ///
    /// Empty shapes have no reachable payload address, so an abstract empty
    /// descriptor is not required to promise alignment for a nonexistent
    /// element. This does not authorize dereferencing a null/unaligned pointer
    /// or constructing an invalid empty Rust slice in a later binding layer.
    /// An unknown cardinality is rejected by callers' existing byte-count
    /// guards; it is not treated as an empty descriptor here.
    #[inline]
    pub fn validate_for_dtype(
        &self, shape: &StaticShape, dtype: DType,
    ) -> Result<(), PlanError> {
        self.validate_for(shape)?;
        if shape.element_count() != Some(0)
            && (self.alignment_bytes as usize) < dtype.size_bytes()
        {
            return Err(PlanError::InvalidTensorAlignment);
        }
        Ok(())
    }

    /// Certified C-order layout test; never reports an invalid public
    /// descriptor as contiguous. Empty and nonempty shapes share the exact
    /// same bounded (rank <= 8) structural validation contract.
    #[inline(always)]
    pub fn is_row_major_contiguous_for(&self, shape: &StaticShape) -> bool {
        self.kind == LayoutKind::RowMajorContiguous
            && self.validate_for(shape).is_ok()
    }

    /// Column-major query with exactly the same strict metadata checks as
    /// the authoritative layout validator, including empty and singleton axes.
    #[inline(always)]
    pub fn is_column_major_contiguous_for(&self, shape: &StaticShape) -> bool {
        self.kind == LayoutKind::ColumnMajorContiguous
            && self.validate_for(shape).is_ok()
    }

    #[inline(always)]
    pub fn is_channels_last_contiguous_for(&self, shape: &StaticShape) -> bool {
        self.kind == LayoutKind::ChannelsLast && self.validate_for(shape).is_ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AccuracyContract {
    Exact,
    StrictFp32,
    /// Allows algebraic floating-point operations/reassociation when supported by the selected toolchain.
    AllowAlgebraic,
    /// Allows target/compiler fast-math transformations such as FTZ or approximate div/sqrt.
    AllowFastMath,
    AllowTf32,
    AllowFp16Accumulate,
    AllowBf16Accumulate,
    QuantizedExactScale,
    TargetVerified,
}

/// Exact numerical mode frozen for a selected kernel artifact. The output tensor's
/// AccuracyContract must permit this mode; replay never changes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KernelMathMode {
    Exact,
    StrictFp32,
    AlgebraicFp,
    FastMath,
    Tf32,
    Fp16Accumulate,
    Bf16Accumulate,
    QuantizedExactScale,
    TargetVerified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ReusePattern {
    SingleUse,
    ReusedWithinReplay,
    ReusedAcrossReplays,
    PersistentWeight,
    PersistentState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SparsityPattern {
    Dense,
    /// One nonzero out of each pair when the selected target-native sparse MMA supports it.
    Structured1To2,
    /// Two nonzeros out of each group of four (classic NVIDIA structured sparsity).
    Structured2To4,
    /// Four nonzeros out of each group of eight for target-native sparse formats that support it.
    Structured4To8,
    BlockSparse,
    StaticUnstructured,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SparsitySpec {
    pub pattern: SparsityPattern,
    pub block_m: u16,
    pub block_n: u16,
    /// Optional tensor carrying indices/masks/metadata when the native packed
    /// representation does not embed them in the same physical storage.
    pub metadata_tensor: Option<TensorId>,
}

impl SparsitySpec {
    /// One authoritative release-scope check for both the canonical graph
    /// and its immutable FrozenExecutionPlan. Future sparse representations
    /// remain wire-schema vocabulary ONLY, not executable support in the
    /// rank-0..8, f32, DENSE_ONLY release.
    #[inline]
    pub fn validate_dense_only(self) -> Result<(), PlanError> {
        if self.pattern != SparsityPattern::Dense
            || self.block_m != 0
            || self.block_n != 0
            || self.metadata_tensor.is_some()
        {
            return Err(PlanError::InvalidSparsitySpec);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuantizationKind {
    None,
    Symmetric,
    Asymmetric,
    MxBlockScaled,
    NvFp4,
    CustomStatic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScaleGranularity {
    None,
    PerTensor,
    PerAxis,
    Block1D,
    Block2D,
}

/// Static quantization/scaling semantics. Scale/zero-point tensors are ordinary
/// tensors in the same canonical graph, so their residency and transfers are
/// planned exactly like any other input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuantizationSpec {
    pub kind: QuantizationKind,
    pub granularity: ScaleGranularity,
    pub axis0: u8,
    pub axis1: u8,
    pub block0: u16,
    pub block1: u16,
    pub scale_tensor: Option<TensorId>,
    pub secondary_scale_tensor: Option<TensorId>,
    pub zero_point_tensor: Option<TensorId>,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorRole {
    /// Ordinary tensor value. In tensor_light this is always F32.
    Data,
    /// Integer metadata consumed by indexing operations.
    Index,
    /// Byte mask metadata.
    Mask,
    /// Canonical unsigned 64-bit auxiliary INDEX_RESULT (arg/sort/topk output).
    /// Not a general-purpose integer Tensor.
    IndexResult,
    /// F32 scale/state/auxiliary tensor.
    Auxiliary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericalTolerance {
    /// Exact IEEE-754 bit pattern of a finite, non-negative absolute error bound.
    pub max_abs_error_f32_bits: u32,
    /// Exact IEEE-754 bit pattern of a finite, non-negative relative error bound.
    pub max_rel_error_f32_bits: u32,
}

impl NumericalTolerance {
    #[inline]
    pub fn is_valid(self) -> bool {
        let abs = f32::from_bits(self.max_abs_error_f32_bits);
        let rel = f32::from_bits(self.max_rel_error_f32_bits);
        abs.is_finite() && rel.is_finite() && abs >= 0.0 && rel >= 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub id: TensorId,
    pub shape: StaticShape,
    pub dtype: DType,
    pub role: TensorRole,
    pub layout: TensorLayout,
    pub accuracy: AccuracyContract,
    pub tolerance: NumericalTolerance,
    pub reuse: ReusePattern,
    pub sparsity: SparsitySpec,
    pub quantization: QuantizationSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OpCode {
    Copy,
    Cast,
    Reshape,
    Transpose,
    Expand,
    AsStrided,
    Narrow,
    Flip,
    Concat,
    Stack,
    Pad,
    Resample2d,
    SlidingWindow1d,
    ZeroInsert1d,
    Tile,
    RepeatInterleave,
    Elementwise,
    ScalarElementwise,
    Select,
    Reduction,
    MatMul,
    FourierTransform,
    DenseLinearAlgebra,
    TensorContract,
    Convolution,
    Attention,
    Collective,
    Normalization,
    Gather,
    Scatter,
    Pooling,
    Softmax,
    Scan,
    Sort,
    TopK,
    Custom,
}



/// Fixed-size axis list used by canonical operation descriptors.
/// Only the first `len` entries are meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisList {
    pub len: u8,
    pub axes: [u8; MAX_RANK],
}

impl AxisList {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self { len: 0, axes: [0; MAX_RANK] }
    }

    pub fn validate(self) -> Result<(), PlanError> {
        let len = usize::from(self.len);
        if len > MAX_RANK {
            return Err(PlanError::InvalidOperationSpec);
        }
        let mut i = 0usize;
        while i < len {
            if usize::from(self.axes[i]) >= MAX_RANK {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut j = i + 1;
            while j < len {
                if self.axes[i] == self.axes[j] {
                    return Err(PlanError::InvalidOperationSpec);
                }
                j += 1;
            }
            i += 1;
        }
        // Entries after `len` are semantically irrelevant. Require their
        // canonical zero representation so equal axis lists cannot hash to
        // different workload identities because of unused tail bytes.
        while i < MAX_RANK {
            if self.axes[i] != 0 {
                return Err(PlanError::InvalidOperationSpec);
            }
            i += 1;
        }
        Ok(())
    }

    pub fn validate_for_rank(self, rank: usize) -> Result<(), PlanError> {
        self.validate()?;
        let mut i = 0usize;
        while i < usize::from(self.len) {
            if usize::from(self.axes[i]) >= rank {
                return Err(PlanError::InvalidOperationSpec);
            }
            i += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisMask(pub u8);

impl AxisMask {
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn validate_for_rank(self, rank: usize) -> Result<(), PlanError> {
        if rank > MAX_RANK {
            return Err(PlanError::InvalidOperationSpec);
        }
        let valid = if rank == MAX_RANK {
            u8::MAX
        } else if rank == 0 {
            0
        } else {
            (1u8 << rank) - 1
        };
        if self.0 & !valid != 0 {
            return Err(PlanError::InvalidOperationSpec);
        }
        Ok(())
    }
}

/// Immutable semantic selection of reduced axes. `All` is not a synonym
/// for an explicit empty set, including at rank zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisSelector {
    All,
    Explicit(AxisMask),
}

impl AxisSelector {
    pub fn is_explicit_empty(self) -> bool {
        matches!(self, Self::Explicit(mask) if mask.is_empty())
    }

    pub fn selection_mask(self, rank: usize) -> Result<u8, PlanError> {
        if rank > MAX_RANK { return Err(PlanError::InvalidOperationSpec); }
        match self {
            Self::All => Ok(if rank == 8 { u8::MAX } else { ((1u16 << rank) - 1) as u8 }),
            Self::Explicit(mask) => {
                mask.validate_for_rank(rank)?;
                Ok(mask.0)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementwiseKind {
    Add,
    Subtract,
    Multiply,
    /// IEEE-754 division. The legacy zero-to-zero compatibility behavior is
    /// represented separately as `DivideZeroToZero` so canonical semantics are exact.
    Divide,
    /// Compatibility divide used by the supplied CPU/GPU kernels: `x / 0 -> 0`.
    DivideZeroToZero,
    /// Canonical minimum: propagate NaN as f32 bit pattern 0x7fc00000;
    /// negative zero sorts below positive zero.
    Minimum,
    /// Canonical maximum: propagate NaN as f32 bit pattern 0x7fc00000;
    /// positive zero sorts above negative zero.
    Maximum,
    Hypot,
    Negate,
    Absolute,
    Reciprocal,
    /// Compatibility reciprocal: `1 / 0 -> 0`.
    ReciprocalZeroToZero,
    Square,
    Sqrt,
    /// Compatibility square root: values `<= 0` map to zero.
    SqrtNonPositiveToZero,
    Rsqrt,
    /// Compatibility reciprocal-square-root: values `<= 0` map to zero.
    RsqrtNonPositiveToZero,
    Exp,
    Expm1,
    Log,
    /// Compatibility logarithm: values `<= 0` map to exactly `-87.0f32`.
    LogNonPositiveToNeg87,
    Log1p,
    /// Compatibility log1p: values `<= -1` map to exactly `-87.0f32`.
    Log1pBelowMinusOneToNeg87,
    Log1pAbs,
    Sin,
    Cos,
    Tan,
    Erf,
    Tanh,
    /// Logistic sigmoid with algebraically stable positive/negative branches;
    /// preserves NaN and gives +0 for -infinity and 1 for +infinity.
    Sigmoid,
    /// ReLU maps negatives and signed zeros to +0; propagates NaN.
    Relu,
    /// ReLU clipped to [0, 6]; propagates NaN and maps signed zero to +0.
    Relu6,
    /// Tanh GELU approximation: 0.5*x*(1 + tanh(0.7978846*(x +
    /// 0.044715*x^3))). This is a distinct f32 mathematical contract, NOT
    /// the erf-based GELU. The -infinity endpoint is defined to be -0.0;
    /// +infinity yields +infinity, and NaNs propagate.
    GeluTanhApprox,
    /// erf-based GELU definition.
    GeluErfExact,
    /// Ambiguous legacy GELU tag. Rejected by validation; use one of the two exact variants.
    Gelu,
    /// x * sigmoid(x), with the -infinity limit defined as -0.
    Silu,
    /// log(1 + exp(x)), evaluated with overflow-safe log1p/exponential form.
    Softplus,
    /// x / (1 + abs(x)); signed infinities map to signed one.
    Softsign,
    /// x * clamp((x+3)/6, 0, 1), with x <= -3 mapped to +0.
    HardSwish,
    /// x * tanh(softplus(x)), with the -infinity limit defined as -0.
    Mish,
    /// ELU with fixed alpha=1: x for x>=0 and expm1(x) for x<0.
    /// NaNs propagate, negative infinity maps to -1, and signed zero is preserved.
    Elu,
    /// SELU with canonical binary32 alpha bits 0x3fd62d7d and scale bits
    /// 0x3f867d5f: scale*x for x>=0, scale*(alpha*expm1(x)) for x<0.
    /// The constants are part of the frozen operation identity, not runtime
    /// configuration or a separate dtype/shape-changing parameter tensor.
    Selu,
    Floor,
    Ceil,
    /// FP32 rounding to nearest integer, ties away from zero (Rust f32::round).
    /// Signed zero and infinities are preserved; NaNs remain NaN.
    Round,
    /// FP32 sign with NaN propagated, signed zero preserved, and finite
    /// nonzero/infinite values mapped to exactly -1 or +1. This is not
    /// Rust f32::signum for zero: signum(-0) returns -1, whereas Sign(-0)
    /// remains -0 under the frozen tensor numeric contract.
    Sign,
    /// Reserved legacy tag: bounds are not encoded here. Use ScalarElementwise::Clamp.
    Clamp,
    CompareEqual,
    CompareNotEqual,
    CompareLess,
    CompareLessEqual,
    CompareGreater,
    CompareGreaterEqual,
    /// Binary PReLU: `x >= 0 ? x : alpha * x`; alpha may broadcast.
    PRelu,
    /// Logical operations consume/produce U8 mask tensors. Any non-zero input byte
    /// is true and outputs are canonicalized to exactly 0 or 1.
    LogicalAnd,
    LogicalOr,
    LogicalNot,
    IsFinite,
    IsNaN,
    IsInf,
    Custom(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarElementwiseKind {
    Add,
    Multiply,
    /// Legacy-compatible scalar power with the exact exponent frozen in scalar0.
    /// Semantics are intentionally piecewise and backend-independent: p=1 is
    /// identity, p=2 is square, p=0.5 is safe sqrt, p=-1 is safe reciprocal,
    /// p=1.5 is x*sqrt(x) for x>0 else 0, and other exponents use
    /// exp(p*log(x)) for x>0 else 0.
    Power,
    /// Scalar division with legacy compatibility semantics: scalar zero produces zero.
    DivideZeroToZero,
    LeakyRelu,
    /// Inclusive clamp to `[scalar0, scalar1]`.
    Clamp,
}

/// Elementwise operation with scalar immediates frozen into the canonical
/// workload. This avoids materializing rank-0 parameter tensors while keeping
/// the exact constants in the plan identity and offline verification contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarElementwiseSpec {
    pub kind: ScalarElementwiseKind,
    pub scalar0_f32_bits: u32,
    /// Must be canonical zero for one-parameter kinds.
    pub scalar1_f32_bits: u32,
}

impl ScalarElementwiseSpec {
    pub fn validate(self) -> Result<(), PlanError> {
        let scalar0 = f32::from_bits(self.scalar0_f32_bits);
        let scalar1 = f32::from_bits(self.scalar1_f32_bits);
        match self.kind {
            ScalarElementwiseKind::Add | ScalarElementwiseKind::Multiply => {
                if self.scalar1_f32_bits != 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
            }
            ScalarElementwiseKind::Power
            | ScalarElementwiseKind::DivideZeroToZero
            | ScalarElementwiseKind::LeakyRelu => {
                if self.scalar1_f32_bits != 0 || !scalar0.is_finite() {
                    return Err(PlanError::InvalidOperationSpec);
                }
            }
            ScalarElementwiseKind::Clamp => {
                if scalar0.is_nan() || scalar1.is_nan() || scalar0 > scalar1 {
                    return Err(PlanError::InvalidOperationSpec);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionKind {
    Sum,
    Product,
    /// Canonical minimum: propagate canonical NaN if any; -0 precedes +0.
    /// Native combine tree and first C-order tie policy are frozen offline.
    Minimum,
    /// Canonical maximum: propagate canonical NaN if any; +0 follows -0.
    /// Native combine tree and first C-order tie policy are frozen offline.
    Maximum,
    Mean,
    /// Population variance: divide by N, not N-1.
    VariancePopulation,
    /// Population standard deviation: sqrt(population variance).
    StdDevPopulation,
    /// Stable mathematical log(sum(exp(x))) reduction. Kernel schedule is target-specific.
    LogSumExp,
    L1Norm,
    L2Norm,
    LInfNorm,
    /// Number of non-zero input values, represented as F32 in tensor_light.
    CountNonZero,
    /// One-axis or all-axes argmin: first NaN wins; otherwise lowest value
    /// (-0 before +0), then first logical C-order tie. Output uses u64.
    ArgMin,
    /// One-axis or all-axes argmax: first NaN wins; otherwise highest value
    /// (+0 after -0), then first logical C-order tie. Output uses u64.
    ArgMax,
    /// Boolean reduction over U8 mask tensors; non-zero is true, output is 0/1.
    Any,
    /// Boolean reduction over U8 mask tensors; non-zero is true, output is 0/1.
    All,
    /// True when any F32 input is non-zero.
    AnyNonZero,
    /// True when every F32 input is non-zero.
    AllNonZero,
    /// True when any F32 input is strictly greater than zero.
    AnyPositive,
    /// True when every F32 input is strictly greater than zero.
    AllPositive,
}


/// The exact semantic identity for an *evaluated* empty reduction slice.
/// This is a typed scalar/auxiliary-mask contract, NOT another Tensor dtype
/// or permission to execute a bulk reduction outside a frozen UASDEA plan.
/// `F32Bits` preserves the exact +0.0/1.0 bit patterns; `MaskByte` uses
/// the canonical unpacked predicate byte representation (0 or 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmptyReductionIdentity {
    F32Bits(u32),
    MaskByte(u8),
}

impl ReductionKind {
    /// The default empty-slice identity is part of the immutable logical
    /// operation contract. Only these operations have published identities;
    /// all other reductions must reject an *evaluated* empty slice until a
    /// distinct, explicitly versioned contract specifies otherwise.
    /// A zero-element OUTPUT does not evaluate any reduction slice.
    #[inline(always)]
    pub const fn empty_reduction_identity(self) -> Option<EmptyReductionIdentity> {
        match self {
            Self::Sum | Self::CountNonZero =>
                Some(EmptyReductionIdentity::F32Bits(0x0000_0000)), // +0.0f32
            Self::Product =>
                Some(EmptyReductionIdentity::F32Bits(0x3f80_0000)), // +1.0f32
            Self::Any | Self::AnyNonZero | Self::AnyPositive =>
                Some(EmptyReductionIdentity::MaskByte(0)), // logical false
            Self::All | Self::AllNonZero | Self::AllPositive =>
                Some(EmptyReductionIdentity::MaskByte(1)), // logical true
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvolutionSpec {
    /// Canonical logical layout is `[..., C, spatial...]`. Weights are
    /// `[O, I/groups, kernel...]` for forward convolution and
    /// `[I, O/groups, kernel...]` for transposed convolution.
    pub spatial_rank: u8,
    pub kernel: [u32; 3],
    pub stride: [u32; 3],
    pub dilation: [u32; 3],
    pub padding_before: [u32; 3],
    pub padding_after: [u32; 3],
    pub groups: u32,
    pub transposed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionMaskKind {
    None,
    /// Query position `q` may attend only key positions `k <= q`.
    Causal,
    /// U8 mask: non-zero means allowed, zero means masked.
    Explicit,
    /// U8 block-sparse mask: non-zero means the block is present/allowed.
    BlockSparse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionSpec {
    /// Canonical Q/K/V logical shape is `[..., H, S, D]`. The output has the
    /// exact query shape; an explicit/block-sparse mask broadcasts to
    /// `[..., query_heads, query_sequence, key_sequence]`.
    pub query_heads: u16,
    pub kv_heads: u16,
    pub head_dim: u32,
    /// Exact IEEE-754 bit pattern of the scale used by the frozen operation.
    pub scale_f32_bits: u32,
    pub mask: AttentionMaskKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    AllReduce,
    ReduceScatter,
    AllGather,
    Broadcast,
    Reduce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveSpec {
    pub kind: CollectiveKind,
    pub group_signature: u64,
    pub root_rank: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormalizationKind {
    /// Normalize over `axes` with population variance: (x-mean)/sqrt(var+epsilon).
    LayerNorm,
    /// x/sqrt(mean(x^2)+epsilon) over `axes`.
    RmsNorm,
    /// Per-channel population statistics over every axis in `axes`.
    /// Affine weight/bias remain separate canonical elementwise nodes and may be fused physically.
    BatchNorm,
    /// Per-sample grouped channel normalization. `axes` contains channel + spatial axes.
    GroupNorm,
    /// Per-sample, per-channel normalization over spatial axes. `axes` excludes
    /// both the sample axis (0) and `channel_axis`; semantically GroupNorm with groups=C.
    InstanceNorm,
    /// x/sqrt(sum(x^2)+epsilon) over `axes`.
    L2Norm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NormalizationSpec {
    pub kind: NormalizationKind,
    pub axes: AxisMask,
    /// Exact IEEE-754 bit pattern; avoids non-canonical float equality in plan identity.
    pub epsilon_f32_bits: u32,
    /// Group count for GroupNorm; canonical zero for every other normalization kind.
    pub groups: u32,
    /// Logical channel axis for BatchNorm/GroupNorm/InstanceNorm; canonical zero otherwise.
    pub channel_axis: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScatterReduction {
    /// Repeated indices are applied in increasing logical index-tensor order;
    /// later values overwrite earlier values deterministically.
    Assign,
    /// Repeated indices reduce in increasing logical index-tensor order.
    Add,
    /// Repeated indices reduce in increasing logical index-tensor order.
    Multiply,
    /// Repeated indices reduce in increasing logical index-tensor order.
    Minimum,
    /// Repeated indices reduce in increasing logical index-tensor order.
    Maximum,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScatterSpec {
    pub axis: u8,
    pub reduction: ScatterReduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolingKind {
    /// Values outside the logical input introduced by padding are negative infinity.
    Maximum,
    /// The divisor contains only valid input elements; padded positions are excluded.
    Average,
    Lp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolingSpec {
    /// Pooling operates on the final `spatial_rank` dimensions and preserves
    /// every leading logical dimension exactly.
    pub kind: PoolingKind,
    pub spatial_rank: u8,
    pub kernel: [u32; 3],
    pub stride: [u32; 3],
    pub dilation: [u32; 3],
    pub padding_before: [u32; 3],
    pub padding_after: [u32; 3],
    pub ceil_mode: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanKind {
    Sum,
    Product,
    Maximum,
    Minimum,
    LogSumExp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanOutput {
    Values,
    /// Valid only for Maximum/Minimum. Ties select the earliest logical index.
    Indices,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScanSpec {
    pub kind: ScanKind,
    pub axis: u8,
    pub inclusive: bool,
    pub output: ScanOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortOutput {
    Values,
    Indices,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortNanPolicy {
    NaNsFirst,
    NaNsLast,
}

/// Canonical one-axis sort/argsort semantics. UASDEA accepts only `stable=true`
/// so equal finite values and equal/NaN groups preserve original logical order.
/// NaN placement is explicit so CPU comparison sort and GPU bitonic/radix-style
/// artifacts cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SortSpec {
    pub axis: u8,
    pub descending: bool,
    pub stable: bool,
    pub output: SortOutput,
    pub nan_policy: SortNanPolicy,
}

/// Canonical top-k selection. Outputs are always `[values, indices]`; values
/// are ordered best-to-worst and equal values preserve the earliest logical
/// source index. NaN placement is explicit so CPU/GPU selections agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopKSpec {
    pub axis: u8,
    pub k: u32,
    pub largest: bool,
    pub nan_policy: SortNanPolicy,
}

/// One-dimensional Fourier transform represented entirely with F32 tensors.
/// Complex tensors are canonicalized as separate real/imaginary F32 tensors so
/// tensor_light remains F32-only while still admitting FFT workloads. Multi-axis
/// FFTs lower to a static sequence of these exact 1-D nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FourierTransformKind {
    ComplexForward,
    ComplexInverse,
    RealForward,
    RealInverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FourierNormalization {
    /// Forward is unscaled; inverse is scaled by 1/N.
    Backward,
    /// Forward is scaled by 1/N; inverse is unscaled.
    Forward,
    /// Both directions are scaled by 1/sqrt(N).
    Ortho,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FourierTransformSpec {
    pub kind: FourierTransformKind,
    pub axis: u8,
    pub normalization: FourierNormalization,
}

/// Dense square linear-algebra kernels recovered from the supplied optimized
/// source. These remain ordinary semantic nodes in the one UASDEA dataflow;
/// CPU, GPU and vendor-library implementations are only alternative frozen
/// mappings selected offline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DenseLinalgKind {
    Solve,
    Inverse,
    CholeskyLower,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinalgFailurePolicy {
    /// Singular / non-SPD batches produce exact +0.0 in the affected output.
    ZeroFill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseLinalgSpec {
    pub kind: DenseLinalgKind,
    pub failure_policy: LinalgFailurePolicy,
    /// Exact finite non-negative threshold used by the frozen validity/pivot
    /// contract. Encoding it in the semantic identity prevents CPU and GPU
    /// implementations from silently disagreeing near singularity.
    pub singular_epsilon_f32_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PadMode {
    Constant,
    Replicate,
    Reflect,
    Circular,
}

/// Rank-preserving N-D padding. Only the first input-rank entries are semantic;
/// all tail entries must remain zero for one canonical encoding. Constant mode
/// preserves the exact IEEE-754 bit pattern, including NaN payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PadSpec {
    pub before: [u32; MAX_RANK],
    pub after: [u32; MAX_RANK],
    pub mode: PadMode,
    pub constant_f32_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResampleMode {
    Nearest,
    /// Legacy-compatible bilinear mapping: src = out * (in - 1) / out_extent.
    Bilinear,
    /// Non-overlapping integer windows with floor-truncated output extents.
    MeanDownsample,
}

/// Integer-scale resampling of the last two logical axes. AOT candidate
/// generation may specialize scale=2 and contiguous spatial layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resample2dSpec {
    pub mode: ResampleMode,
    pub scale: u32,
}

/// Exact rank-preserving slice used by narrow/split/chunk canonicalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NarrowSpec {
    pub axis: u8,
    pub start: u64,
    pub length: u64,
}

/// Storage-exposing read-only view. The output tensor's canonical `Strided`
/// layout carries the exact element strides; this value shifts logical output
/// element zero relative to the input view's logical origin. Frozen mappings
/// may realize it only when the source physical representation preserves the
/// semantic backing-storage order, so CPU and GPU cannot silently disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AsStridedSpec {
    pub storage_offset_elements: isize,
}

/// One-dimensional sliding window. The canonical output is rank-2 with
/// shape [rows, window], where rows = 1 + (n-window)/step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlidingWindow1dSpec {
    pub window: u32,
    pub step: u32,
}

/// Per-axis periodic repetition. Only the first `rank` entries are semantic;
/// tail entries must be one so equivalent workloads have one identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileSpec {
    pub rank: u8,
    pub repeats: [u32; MAX_RANK],
}

/// Repeat each element `repeats` times along one axis. Unlike Tile, repeated
/// values are consecutive rather than periodic over the whole source extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RepeatInterleaveSpec {
    pub axis: u8,
    pub repeats: u32,
}

/// Canonical semantic operation descriptor. Performance mappings belong to the
/// frozen CPU/GPU placement plans, not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Copy,
    /// Reserved until exact rounding/saturation semantics are represented in
    /// the canonical IR. `tensor_light` rejects this operation today rather
    /// than allowing backend-dependent conversion behavior.
    Cast,
    Reshape,
    Transpose { permutation: AxisList },
    /// Standard right-aligned broadcast/expand. A frozen mapping may keep this
    /// as a zero-stride alias instead of materializing.
    Expand,
    AsStrided(AsStridedSpec),
    /// Rank-preserving contiguous logical slice. Split/chunk lower to one or
    /// more Narrow nodes so the execution plan can preserve zero-copy aliases.
    Narrow(NarrowSpec),
    /// Reverse one or more axes. Physical mappings may encode this with signed
    /// strides and a shifted base offset, or benchmark a materialized kernel.
    Flip { axes: AxisList },
    /// Concatenate same-rank tensors along one existing axis.
    Concat { axis: u8 },
    /// Insert a new axis and stack equal-shaped inputs along it.
    Stack { axis: u8 },
    Pad(PadSpec),
    Resample2d(Resample2dSpec),
    SlidingWindow1d(SlidingWindow1dSpec),
    /// Rank-1 zero-insertion dilation. For input length n > 1 and factor f > 1,
    /// output length is (n - 1) * f + 1; source samples occupy indices i*f
    /// and inserted positions are exact +0.0f32. Empty inputs produce empty
    /// output, and n=1 or f=1 are semantic identities. Every case still
    /// requires an explicitly published frozen graph/alias/copy effect.
    ZeroInsert1d { factor: u32 },
    Tile(TileSpec),
    RepeatInterleave(RepeatInterleaveSpec),
    Elementwise { kind: ElementwiseKind, arity: u8, broadcast_signature: u64 },
    ScalarElementwise(ScalarElementwiseSpec),
    /// Ternary mask select with standard trailing-axis broadcasting. Inputs are
    /// `[condition_mask, true_value, false_value]`; any non-zero mask byte selects
    /// the true input.
    Select { broadcast_signature: u64 },
    Reduction { kind: ReductionKind, axes: AxisSelector, keep_dims: bool },
    /// NumPy-style matmul with static broadcasting over leading batch axes.
    /// Rank-1 operands use vector semantics: 1D x 1D -> scalar, 1D x ND ->
    /// the promoted leading matrix dimension is removed, and ND x 1D -> the
    /// promoted trailing matrix dimension is removed. Transpose flags are
    /// invalid for the corresponding rank-1 operand.
    MatMul { transpose_lhs: bool, transpose_rhs: bool },
    FourierTransform(FourierTransformSpec),
    DenseLinearAlgebra(DenseLinalgSpec),
    TensorContract { lhs_axes: AxisList, rhs_axes: AxisList },
    Convolution(ConvolutionSpec),
    Attention(AttentionSpec),
    Collective(CollectiveSpec),
    Normalization(NormalizationSpec),
    /// Indices are zero-based, non-negative and must be in range on every
    /// successful replay. Kernels for this operation may not rely on
    /// `unchecked_indexing` because index values are runtime data.
    Gather { axis: u8 },
    /// Uses the same zero-based/in-range index contract as Gather. A rank-0
    /// values tensor is broadcast over every selected element; otherwise the
    /// values shape equals the base shape with the scatter axis replaced by
    /// the index-vector length. Repeated indices follow `ScatterReduction`'s
    /// deterministic logical order.
    Scatter(ScatterSpec),
    Pooling(PoolingSpec),
    /// Softmax or log-softmax over one exact logical axis.
    Softmax { axis: u8, log: bool },
    Scan(ScanSpec),
    Sort(SortSpec),
    TopK(TopKSpec),
    Custom { semantic_signature_hi: u64, semantic_signature_lo: u64 },
}

impl Operation {
    #[inline(always)]
    pub const fn opcode(self) -> OpCode {
        match self {
            Self::Copy => OpCode::Copy,
            Self::Cast => OpCode::Cast,
            Self::Reshape => OpCode::Reshape,
            Self::Transpose { .. } => OpCode::Transpose,
            Self::Expand => OpCode::Expand,
            Self::AsStrided(_) => OpCode::AsStrided,
            Self::Narrow(_) => OpCode::Narrow,
            Self::Flip { .. } => OpCode::Flip,
            Self::Concat { .. } => OpCode::Concat,
            Self::Stack { .. } => OpCode::Stack,
            Self::Pad(_) => OpCode::Pad,
            Self::Resample2d(_) => OpCode::Resample2d,
            Self::SlidingWindow1d(_) => OpCode::SlidingWindow1d,
            Self::ZeroInsert1d { .. } => OpCode::ZeroInsert1d,
            Self::Tile(_) => OpCode::Tile,
            Self::RepeatInterleave(_) => OpCode::RepeatInterleave,
            Self::Elementwise { .. } => OpCode::Elementwise,
            Self::ScalarElementwise(_) => OpCode::ScalarElementwise,
            Self::Select { .. } => OpCode::Select,
            Self::Reduction { .. } => OpCode::Reduction,
            Self::MatMul { .. } => OpCode::MatMul,
            Self::FourierTransform(_) => OpCode::FourierTransform,
            Self::DenseLinearAlgebra(_) => OpCode::DenseLinearAlgebra,
            Self::TensorContract { .. } => OpCode::TensorContract,
            Self::Convolution(_) => OpCode::Convolution,
            Self::Attention(_) => OpCode::Attention,
            Self::Collective(_) => OpCode::Collective,
            Self::Normalization(_) => OpCode::Normalization,
            Self::Gather { .. } => OpCode::Gather,
            Self::Scatter(_) => OpCode::Scatter,
            Self::Pooling(_) => OpCode::Pooling,
            Self::Softmax { .. } => OpCode::Softmax,
            Self::Scan(_) => OpCode::Scan,
            Self::Sort(_) => OpCode::Sort,
            Self::TopK(_) => OpCode::TopK,
            Self::Custom { .. } => OpCode::Custom,
        }
    }

    pub fn validate_basic(self) -> Result<(), PlanError> {
        match self {
            Self::Transpose { permutation } => permutation.validate(),
            Self::Expand => Ok(()),
            Self::AsStrided(_) => Ok(()),
            Self::Narrow(spec) => {
                if usize::from(spec.axis) >= MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Flip { axes } => {
                axes.validate()?;
                if axes.len == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Concat { axis } => {
                if usize::from(axis) >= MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Stack { axis } => {
                if usize::from(axis) > MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Pad(spec) => {
                if spec.mode != PadMode::Constant && spec.constant_f32_bits != 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Resample2d(spec) => {
                if spec.scale == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::SlidingWindow1d(spec) => {
                if spec.window == 0 || spec.step == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::ZeroInsert1d { factor } => {
                // factor=1 is a defined identity, requiring a published graph
                // output effect just like any other graph operation.
                if factor == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Tile(spec) => {
                let rank = usize::from(spec.rank);
                // A scalar is rank zero with one logical f32 element. Its
                // zero-axis Tile with all inactive repeats set to one is a
                // legal identity, just as zero-width Pad is on rank zero.
                // Any repetition introducing an axis must use a separately
                // published rank-changing operation, not this rank-preserving
                // Tile contract.
                if rank > MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                // An all-one repetition vector is a valid semantic identity.
                // Keep the node in the canonical graph; do not rewrite or
                // materialize an alternative operation at runtime.
                let mut i = rank;
                while i < MAX_RANK {
                    if spec.repeats[i] != 1 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    i += 1;
                }
                Ok(())
            }
            Self::RepeatInterleave(spec) => {
                if usize::from(spec.axis) >= MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                // repeats=1 is valid (bit-preserving identity); repeats=0
                // has the separately checked zero-length output semantics.
                Ok(())
            }
            Self::Elementwise { kind, arity, .. } => {
                let expected_arity = match kind {
                    ElementwiseKind::Negate
                    | ElementwiseKind::Absolute
                    | ElementwiseKind::Reciprocal
                    | ElementwiseKind::ReciprocalZeroToZero
                    | ElementwiseKind::Square
                    | ElementwiseKind::Sqrt
                    | ElementwiseKind::SqrtNonPositiveToZero
                    | ElementwiseKind::Rsqrt
                    | ElementwiseKind::RsqrtNonPositiveToZero
                    | ElementwiseKind::Exp
                    | ElementwiseKind::Expm1
                    | ElementwiseKind::Log
                    | ElementwiseKind::LogNonPositiveToNeg87
                    | ElementwiseKind::Log1p
                    | ElementwiseKind::Log1pBelowMinusOneToNeg87
                    | ElementwiseKind::Log1pAbs
                    | ElementwiseKind::Sin
                    | ElementwiseKind::Cos
                    | ElementwiseKind::Tan
                    | ElementwiseKind::Erf
                    | ElementwiseKind::Tanh
                    | ElementwiseKind::Sigmoid
                    | ElementwiseKind::Relu
                    | ElementwiseKind::Relu6
                    | ElementwiseKind::GeluTanhApprox
                    | ElementwiseKind::GeluErfExact
                    | ElementwiseKind::Silu
                    | ElementwiseKind::Softplus
                    | ElementwiseKind::Softsign
                    | ElementwiseKind::HardSwish
                    | ElementwiseKind::Mish
                    | ElementwiseKind::Elu
                    | ElementwiseKind::Selu
                    | ElementwiseKind::Floor
                    | ElementwiseKind::Ceil
                    | ElementwiseKind::Round
                    | ElementwiseKind::Sign
                    | ElementwiseKind::LogicalNot
                    | ElementwiseKind::IsFinite
                    | ElementwiseKind::IsNaN
                    | ElementwiseKind::IsInf => 1,
                    ElementwiseKind::Add
                    | ElementwiseKind::Subtract
                    | ElementwiseKind::Multiply
                    | ElementwiseKind::Divide
                    | ElementwiseKind::DivideZeroToZero
                    | ElementwiseKind::Minimum
                    | ElementwiseKind::Maximum
                    | ElementwiseKind::Hypot
                    | ElementwiseKind::CompareEqual
                    | ElementwiseKind::CompareNotEqual
                    | ElementwiseKind::CompareLess
                    | ElementwiseKind::CompareLessEqual
                    | ElementwiseKind::CompareGreater
                    | ElementwiseKind::CompareGreaterEqual
                    | ElementwiseKind::PRelu
                    | ElementwiseKind::LogicalAnd
                    | ElementwiseKind::LogicalOr => 2,
                    // Bounds/approximation semantics are ambiguous for these legacy tags.
                    ElementwiseKind::Clamp | ElementwiseKind::Gelu => {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    // A 16-bit custom code does not uniquely define semantics.
                    ElementwiseKind::Custom(_) => {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                };
                if arity != expected_arity {
                    Err(PlanError::InvalidOperationSpec)
                } else {
                    Ok(())
                }
            }
            Self::ScalarElementwise(spec) => spec.validate(),
            Self::Select { .. } => Ok(()),
            Self::Reduction { kind, axes, .. } => {
                if matches!(kind, ReductionKind::ArgMin | ReductionKind::ArgMax)
                    && axes.is_explicit_empty()
                {
                    Err(PlanError::InvalidOperationSpec)
                } else {
                    Ok(())
                }
            }
            Self::FourierTransform(spec) => {
                if usize::from(spec.axis) >= MAX_RANK {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::DenseLinearAlgebra(spec) => {
                let epsilon = f32::from_bits(spec.singular_epsilon_f32_bits);
                if !epsilon.is_finite() || epsilon < 0.0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::TensorContract { lhs_axes, rhs_axes } => {
                lhs_axes.validate()?;
                rhs_axes.validate()?;
                if lhs_axes.len == 0 || lhs_axes.len != rhs_axes.len {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Convolution(spec) => validate_spatial(
                spec.spatial_rank,
                spec.kernel,
                spec.stride,
                spec.dilation,
                spec.padding_before,
                spec.padding_after,
                spec.groups,
            ),
            Self::Attention(spec) => {
                let scale = f32::from_bits(spec.scale_f32_bits);
                if spec.query_heads == 0
                    || spec.kv_heads == 0
                    || spec.head_dim == 0
                    || spec.query_heads % spec.kv_heads != 0
                    || !scale.is_finite()
                    || scale <= 0.0
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Collective(spec) => {
                if spec.group_signature == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                match spec.kind {
                    CollectiveKind::Broadcast => {
                        if spec.root_rank.is_none() {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                    }
                    // Reduction collectives need an explicit reduction operator;
                    // shape-changing collectives additionally need a group-size/
                    // partition-axis contract. The light IR intentionally rejects
                    // these underspecified forms instead of hashing ambiguous semantics.
                    CollectiveKind::AllReduce
                    | CollectiveKind::Reduce
                    | CollectiveKind::ReduceScatter
                    | CollectiveKind::AllGather => {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                Ok(())
            }
            Self::Normalization(spec) => {
                let epsilon = f32::from_bits(spec.epsilon_f32_bits);
                if spec.axes.is_empty() || !epsilon.is_finite() || epsilon < 0.0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                match spec.kind {
                    NormalizationKind::LayerNorm
                    | NormalizationKind::RmsNorm
                    | NormalizationKind::L2Norm => {
                        if spec.groups != 0 || spec.channel_axis != 0 {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                    }
                    NormalizationKind::BatchNorm | NormalizationKind::InstanceNorm => {
                        if spec.groups != 0 || spec.channel_axis == 0 {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                    }
                    NormalizationKind::GroupNorm => {
                        if spec.groups == 0 || spec.channel_axis == 0 {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                    }
                }
                Ok(())
            }
            Self::Pooling(spec) => {
                // Lp pooling needs an explicit p/order parameter to have exact
                // semantics. This light IR does not carry one, so do not admit
                // an ambiguous frozen plan.
                if spec.kind == PoolingKind::Lp {
                    return Err(PlanError::InvalidOperationSpec);
                }
                validate_spatial(
                    spec.spatial_rank,
                    spec.kernel,
                    spec.stride,
                    spec.dilation,
                    spec.padding_before,
                    spec.padding_after,
                    1,
                )
            }
            Self::Custom { semantic_signature_hi, semantic_signature_lo } => {
                if semantic_signature_hi == 0 && semantic_signature_lo == 0 {
                    Err(PlanError::InvalidOperationSpec)
                } else {
                    Ok(())
                }
            }
            Self::Scan(spec) => {
                if spec.output == ScanOutput::Indices
                    && !matches!(spec.kind, ScanKind::Maximum | ScanKind::Minimum)
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
                // The supplied kernels expose exclusive prefix only for additive scan.
                if !spec.inclusive && spec.kind != ScanKind::Sum {
                    return Err(PlanError::InvalidOperationSpec);
                }
                if spec.output == ScanOutput::Indices && !spec.inclusive {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Sort(spec) => {
                // A non-stable argsort leaves equal-value index order undefined,
                // which is not a canonical UASDEA semantic result. AOT kernels
                // may use any internal algorithm but must reproduce stable order.
                if !spec.stable {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::TopK(spec) => {
                if spec.k == 0 {
                    return Err(PlanError::InvalidOperationSpec);
                }
                Ok(())
            }
            Self::Cast => Err(PlanError::InvalidOperationSpec),
            // SECURITY/SEMANTICS: v24 requires a sealed, full INDEX_INPUT value
            // validation stage and an ordered error gate BEFORE any native
            // unchecked gather load or scatter store. This light IR currently
            // has neither certified stage nor failure/commit dependency. Shape
            // checks alone never authorize a dynamic-index native kernel.
            // Reject the entire operation rather than accepting unsafe plans.
            // Re-enable ONLY alongside a versioned, AOT-frozen validator and
            // corresponding CPU/GPU/partial-submission error tests.
            Self::Gather { .. } | Self::Scatter(_) => Err(PlanError::UnsealedIndexPreflight),
            Self::Copy
            | Self::Reshape
            | Self::MatMul { .. }
            | Self::Softmax { .. } => Ok(()),
        }
    }
}

fn validate_spatial(
    spatial_rank: u8,
    kernel: [u32; 3],
    stride: [u32; 3],
    dilation: [u32; 3],
    padding_before: [u32; 3],
    padding_after: [u32; 3],
    groups: u32,
) -> Result<(), PlanError> {
    if spatial_rank == 0 || spatial_rank > 3 || groups == 0 {
        return Err(PlanError::InvalidOperationSpec);
    }
    let rank = usize::from(spatial_rank);
    let mut axis = 0usize;
    while axis < rank {
        if kernel[axis] == 0 || stride[axis] == 0 || dilation[axis] == 0 {
            return Err(PlanError::InvalidOperationSpec);
        }
        axis += 1;
    }
    // The fixed arrays have three entries, but only `spatial_rank` are semantic.
    // Freeze neutral values in the unused tail so the canonical workload hash
    // has a single representation for each operation.
    while axis < 3 {
        if kernel[axis] != 1
            || stride[axis] != 1
            || dilation[axis] != 1
            || padding_before[axis] != 0
            || padding_after[axis] != 0
        {
            return Err(PlanError::InvalidOperationSpec);
        }
        axis += 1;
    }
    Ok(())
}


// =============================================================================
// Canonical workload and exact identity
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalNode {
    pub id: NodeId,
    pub operation: Operation,
    pub dependencies: &'static [NodeId],
    pub inputs: &'static [TensorId],
    pub outputs: &'static [TensorId],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalStaticDataflow {
    pub tensors: &'static [TensorSpec],
    /// Nodes are stored in canonical topological order.
    pub nodes: &'static [CanonicalNode],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PeerAccessPair {
    pub src_device: u16,
    pub dst_device: u16,
    /// CUDA P2P performance rank; lower means a faster/preferred path.
    pub performance_rank: u32,
    pub native_atomic_supported: bool,
    pub partial_native_atomic_supported: bool,
    pub cuda_array_access_supported: bool,
    /// Canonical digest of per-operation P2P atomic capabilities when support is partial.
    /// Zero when partial-native-atomic support is false.
    pub atomic_capabilities_hash: u64,
}

/// Static memory/interconnect topology visible to offline specialization.
/// This is a workload/environment contract, not a selected memory plan.
/// Directed host-NUMA distance used by offline placement/copy cost modelling.
/// The numeric scale is platform-defined (for example Linux SLIT distance), so
/// only relative values inside one frozen hardware profile are meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostNumaDistance {
    pub src_node: u16,
    pub dst_node: u16,
    pub distance: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryTopologyContract {
    pub host_numa_nodes: &'static [u16],
    /// Exact directed NUMA distance matrix in canonical (src,dst) order when
    /// the platform exposes one. Empty means unavailable; otherwise the matrix
    /// is complete, including diagonals.
    pub host_numa_distances: &'static [HostNumaDistance],
    pub gpu_device_ordinals: &'static [u16],
    pub peer_access: &'static [PeerAccessPair],
    pub supports_pinned_host_memory: bool,
    pub supports_mapped_host_memory: bool,
    pub supports_unified_memory: bool,
}

impl MemoryTopologyContract {
    /// Deterministic identity of the exact memory/interconnect capability set.
    /// Runtime capability binding compares this digest in addition to the
    /// hardware-profile digest so P2P/NUMA/host-memory topology cannot drift.
    #[inline]
    pub fn canonical_hash(&self) -> (u64, u64) {
        stable_hash_pair(self, 0x4d45_4d54_4f50_4f4c) // MEMTOPOL
    }
}

/// Resources/topologies that the canonical workload permits the offline stage
/// to map onto. A FrozenExecutionPlan chooses exactly one mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionTopologyContract {
    pub cpu_enabled: bool,
    pub gpu_device_ordinals: &'static [u16],
    pub allow_cpu_gpu_partitioned: bool,
    pub allow_cpu_gpu_pipeline: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkloadSignature {
    pub graph_hi: u64,
    pub graph_lo: u64,
    pub shape_dtype_layout_alignment_hi: u64,
    pub shape_dtype_layout_alignment_lo: u64,
    pub accuracy_reuse_sparsity_hi: u64,
    pub accuracy_reuse_sparsity_lo: u64,
    pub hardware_hi: u64,
    pub hardware_lo: u64,
    pub memory_topology_hi: u64,
    pub memory_topology_lo: u64,
    pub execution_topology_hi: u64,
    pub execution_topology_lo: u64,
}

impl WorkloadSignature {
    #[inline(always)]
    pub const fn is_complete(self) -> bool {
        (self.graph_hi | self.graph_lo) != 0
            && (self.shape_dtype_layout_alignment_hi | self.shape_dtype_layout_alignment_lo) != 0
            && (self.accuracy_reuse_sparsity_hi | self.accuracy_reuse_sparsity_lo) != 0
            && (self.hardware_hi | self.hardware_lo) != 0
            && (self.memory_topology_hi | self.memory_topology_lo) != 0
            && (self.execution_topology_hi | self.execution_topology_lo) != 0
    }

    /// Mechanical exact-key derivation. Runtime may use this only after it has
    /// the complete canonical workload; it is never a performance selector.
    pub const fn plan_key(self) -> PlanKey {
        let mut hi = 0x243f_6a88_85a3_08d3u64;
        let mut lo = 0x1319_8a2e_0370_7344u64;
        hi = mix64(hi ^ self.graph_hi);
        lo = mix64(lo ^ self.graph_lo);
        hi = mix64(hi ^ self.shape_dtype_layout_alignment_hi);
        lo = mix64(lo ^ self.shape_dtype_layout_alignment_lo);
        hi = mix64(hi ^ self.accuracy_reuse_sparsity_hi);
        lo = mix64(lo ^ self.accuracy_reuse_sparsity_lo);
        hi = mix64(hi ^ self.hardware_hi);
        lo = mix64(lo ^ self.hardware_lo);
        hi = mix64(hi ^ self.memory_topology_hi);
        lo = mix64(lo ^ self.memory_topology_lo);
        hi = mix64(hi ^ self.execution_topology_hi);
        lo = mix64(lo ^ self.execution_topology_lo);
        PlanKey { signature_hi: hi, signature_lo: lo }
    }
}

const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalWorkload {
    pub graph: CanonicalStaticDataflow,
    pub hardware: HardwareContract,
    pub memory_topology: MemoryTopologyContract,
    pub execution_topology: ExecutionTopologyContract,
}

impl CanonicalWorkload {
    pub fn validate(&self) -> Result<(), PlanError> {
        self.validate_topologies()?;
        self.validate_graph()?;
        Ok(())
    }

    /// Canonical identity that numerical verification must name. This binds a
    /// verification record to the exact semantic graph/tensor contracts rather
    /// than accepting an arbitrary non-zero verifier marker.
    #[inline]
    pub fn verification_reference_signature(&self) -> (u64, u64) {
        stable_hash_pair(&self.graph, 0x5645_5249_4652_4546) // VERIFREF
    }

    pub fn signature(&self) -> WorkloadSignature {
        let graph = stable_hash_pair(&self.graph.nodes, 0x4341_4e4f_4e47_5241); // CANONGRA

        let mut shape_hi = StableIdentityHasher::new(0x5348_4150_4544_5459); // SHAPEDTY
        let mut shape_lo = StableIdentityHasher::new(0x4c41_594f_5554_414c); // LAYOUTAL
        let mut semantics_hi = StableIdentityHasher::new(0x4143_4355_5241_4359); // ACCURACY
        let mut semantics_lo = StableIdentityHasher::new(0x5245_5553_4553_5041); // REUSESPA

        let mut i = 0usize;
        while i < self.graph.tensors.len() {
            let tensor = &self.graph.tensors[i];
            tensor.id.hash(&mut shape_hi);
            tensor.id.hash(&mut shape_lo);
            tensor.shape.hash(&mut shape_hi);
            tensor.shape.hash(&mut shape_lo);
            tensor.dtype.hash(&mut shape_hi);
            tensor.dtype.hash(&mut shape_lo);
            tensor.role.hash(&mut shape_hi);
            tensor.role.hash(&mut shape_lo);
            tensor.layout.hash(&mut shape_hi);
            tensor.layout.hash(&mut shape_lo);

            tensor.id.hash(&mut semantics_hi);
            tensor.id.hash(&mut semantics_lo);
            tensor.accuracy.hash(&mut semantics_hi);
            tensor.accuracy.hash(&mut semantics_lo);
            tensor.tolerance.hash(&mut semantics_hi);
            tensor.tolerance.hash(&mut semantics_lo);
            tensor.reuse.hash(&mut semantics_hi);
            tensor.reuse.hash(&mut semantics_lo);
            tensor.sparsity.hash(&mut semantics_hi);
            tensor.sparsity.hash(&mut semantics_lo);
            tensor.quantization.hash(&mut semantics_hi);
            tensor.quantization.hash(&mut semantics_lo);
            i += 1;
        }

        let memory = self.memory_topology.canonical_hash();
        let execution = stable_hash_pair(&self.execution_topology, 0x4558_4543_544f_504f); // EXECTOPO

        let hardware = self.hardware.computed_profile_hash();

        WorkloadSignature {
            graph_hi: graph.0,
            graph_lo: graph.1,
            shape_dtype_layout_alignment_hi: shape_hi.finish(),
            shape_dtype_layout_alignment_lo: shape_lo.finish(),
            accuracy_reuse_sparsity_hi: semantics_hi.finish(),
            accuracy_reuse_sparsity_lo: semantics_lo.finish(),
            hardware_hi: hardware.0,
            hardware_lo: hardware.1,
            memory_topology_hi: memory.0,
            memory_topology_lo: memory.1,
            execution_topology_hi: execution.0,
            execution_topology_lo: execution.1,
        }
    }

    /// Conservative global verification contract. The offline verifier reports
    /// one worst-case absolute/relative pair, so the central pipeline uses the
    /// strictest tolerance of every canonical tensor produced by the graph.
    pub fn verification_tolerance(&self) -> (f32, f32) {
        let mut abs = f32::INFINITY;
        let mut rel = f32::INFINITY;
        let mut found = false;
        let mut n = 0usize;
        while n < self.graph.nodes.len() {
            let node = &self.graph.nodes[n];
            let mut o = 0usize;
            while o < node.outputs.len() {
                if let Some(tensor) = self.find_tensor(node.outputs[o]) {
                    if matches!(tensor.role, TensorRole::Index | TensorRole::IndexResult | TensorRole::Mask) {
                        o += 1;
                        continue;
                    }
                    let ta = f32::from_bits(tensor.tolerance.max_abs_error_f32_bits);
                    let tr = f32::from_bits(tensor.tolerance.max_rel_error_f32_bits);
                    if ta < abs {
                        abs = ta;
                    }
                    if tr < rel {
                        rel = tr;
                    }
                    found = true;
                }
                o += 1;
            }
            n += 1;
        }
        if found { (abs, rel) } else { (0.0, 0.0) }
    }

    #[inline(always)]
    pub fn find_tensor(&self, id: TensorId) -> Option<&TensorSpec> {
        self.graph
            .tensors
            .get(id as usize)
            .filter(|tensor| tensor.id == id)
    }

    #[inline(always)]
    pub fn find_node(&self, id: NodeId) -> Option<&CanonicalNode> {
        self.graph
            .nodes
            .get(id as usize)
            .filter(|node| node.id == id)
    }

    fn validate_topologies(&self) -> Result<(), PlanError> {
        self.hardware.validate_canonical()?;
        if !self.execution_topology.cpu_enabled && self.execution_topology.gpu_device_ordinals.is_empty() {
            return Err(PlanError::InvalidExecutionTopology);
        }
        if self.execution_topology.cpu_enabled && self.hardware.cpu.is_none() {
            return Err(PlanError::InvalidExecutionTopology);
        }
        if self.execution_topology.cpu_enabled && self.memory_topology.host_numa_nodes.is_empty() {
            return Err(PlanError::InvalidMemoryTopology);
        }
        if (self.execution_topology.allow_cpu_gpu_partitioned
            || self.execution_topology.allow_cpu_gpu_pipeline)
            && (!self.execution_topology.cpu_enabled
                || self.execution_topology.gpu_device_ordinals.is_empty())
        {
            return Err(PlanError::InvalidExecutionTopology);
        }

        let mut i = 0usize;
        while i < self.execution_topology.gpu_device_ordinals.len() {
            let ordinal = self.execution_topology.gpu_device_ordinals[i];
            if i > 0 && self.execution_topology.gpu_device_ordinals[i - 1] >= ordinal {
                return Err(PlanError::InvalidExecutionTopology);
            }
            let mut found = false;
            let mut h = 0usize;
            while h < self.hardware.gpus.len() {
                if self.hardware.gpus[h].device_ordinal == ordinal {
                    found = true;
                    break;
                }
                h += 1;
            }
            if !found {
                return Err(PlanError::InvalidExecutionTopology);
            }
            if !self.memory_topology.gpu_device_ordinals.contains(&ordinal) {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let mut j = i + 1;
            while j < self.execution_topology.gpu_device_ordinals.len() {
                if self.execution_topology.gpu_device_ordinals[j] == ordinal {
                    return Err(PlanError::InvalidExecutionTopology);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_topology.gpu_device_ordinals.len() {
            let ordinal = self.memory_topology.gpu_device_ordinals[i];
            if i > 0 && self.memory_topology.gpu_device_ordinals[i - 1] >= ordinal {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let mut found = false;
            let mut h = 0usize;
            while h < self.hardware.gpus.len() {
                if self.hardware.gpus[h].device_ordinal == ordinal {
                    found = true;
                    break;
                }
                h += 1;
            }
            if !found {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let mut j = i + 1;
            while j < self.memory_topology.gpu_device_ordinals.len() {
                if self.memory_topology.gpu_device_ordinals[j] == ordinal {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_topology.host_numa_nodes.len() {
            if i > 0
                && self.memory_topology.host_numa_nodes[i - 1]
                    >= self.memory_topology.host_numa_nodes[i]
            {
                return Err(PlanError::InvalidMemoryTopology);
            }
            if let Some(cpu) = self.hardware.cpu {
                if self.memory_topology.host_numa_nodes[i] >= cpu.numa_node_count {
                    return Err(PlanError::InvalidMemoryTopology);
                }
            } else {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let mut j = i + 1;
            while j < self.memory_topology.host_numa_nodes.len() {
                if self.memory_topology.host_numa_nodes[i] == self.memory_topology.host_numa_nodes[j] {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                j += 1;
            }
            i += 1;
        }

        let numa_count = self.memory_topology.host_numa_nodes.len();
        let expected_numa_distances = numa_count
            .checked_mul(numa_count)
            .ok_or(PlanError::InvalidMemoryTopology)?;
        if !self.memory_topology.host_numa_distances.is_empty()
            && self.memory_topology.host_numa_distances.len() != expected_numa_distances
        {
            return Err(PlanError::InvalidMemoryTopology);
        }
        i = 0;
        while i < self.memory_topology.host_numa_distances.len() {
            let distance = self.memory_topology.host_numa_distances[i];
            if distance.distance == 0
                || !self.memory_topology.host_numa_nodes.contains(&distance.src_node)
                || !self.memory_topology.host_numa_nodes.contains(&distance.dst_node)
            {
                return Err(PlanError::InvalidMemoryTopology);
            }
            if i > 0 {
                let previous = self.memory_topology.host_numa_distances[i - 1];
                if (previous.src_node, previous.dst_node)
                    >= (distance.src_node, distance.dst_node)
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_topology.peer_access.len() {
            let pair = self.memory_topology.peer_access[i];
            if i > 0 {
                let previous = self.memory_topology.peer_access[i - 1];
                if (previous.src_device, previous.dst_device)
                    >= (pair.src_device, pair.dst_device)
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
            }
            if pair.src_device == pair.dst_device
                || !self.memory_topology.gpu_device_ordinals.contains(&pair.src_device)
                || !self.memory_topology.gpu_device_ordinals.contains(&pair.dst_device)
                || (pair.native_atomic_supported && pair.partial_native_atomic_supported)
                || (pair.partial_native_atomic_supported && pair.atomic_capabilities_hash == 0)
                || (!pair.partial_native_atomic_supported && pair.atomic_capabilities_hash != 0)
            {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let src = self
                .hardware
                .gpus
                .iter()
                .find(|gpu| gpu.device_ordinal == pair.src_device)
                .ok_or(PlanError::InvalidMemoryTopology)?;
            let dst = self
                .hardware
                .gpus
                .iter()
                .find(|gpu| gpu.device_ordinal == pair.dst_device)
                .ok_or(PlanError::InvalidMemoryTopology)?;
            if !src.features.contains(gpu_feature::PEER_TO_PEER)
                || !dst.features.contains(gpu_feature::PEER_TO_PEER)
            {
                return Err(PlanError::InvalidMemoryTopology);
            }
            let mut j = i + 1;
            while j < self.memory_topology.peer_access.len() {
                if self.memory_topology.peer_access[j] == pair {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                j += 1;
            }
            i += 1;
        }
        Ok(())
    }

    fn validate_graph(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.graph.tensors.len() {
            let tensor = &self.graph.tensors[i];
            // TensorId is a referential label rather than semantic payload.
            // Contiguous canonical numbering prevents equivalent graphs from
            // receiving different workload identities solely from arbitrary IDs.
            if tensor.id != i as TensorId {
                return Err(PlanError::InvalidCanonicalGraph);
            }
            tensor.layout.validate_for_dtype(&tensor.shape, tensor.dtype)?;
            tensor.shape.byte_count(tensor.dtype).ok_or(PlanError::TensorByteCountOverflow)?;
            if !tensor.tolerance.is_valid() {
                return Err(PlanError::InvalidAccuracyContract);
            }
            let abs = f32::from_bits(tensor.tolerance.max_abs_error_f32_bits);
            let rel = f32::from_bits(tensor.tolerance.max_rel_error_f32_bits);
            if tensor.accuracy == AccuracyContract::Exact && (abs != 0.0 || rel != 0.0) {
                return Err(PlanError::InvalidAccuracyContract);
            }
            if tensor.accuracy == AccuracyContract::QuantizedExactScale {
                // This concrete tensor_light variant has no quantized value
                // tensors, so a quantized numerical contract is not meaningful.
                return Err(PlanError::InvalidAccuracyContract);
            }
            if tensor.reuse == ReusePattern::PersistentState {
                // Stateful replay needs an explicit state-transition/state-version
                // contract. This static SSA IR intentionally does not infer one.
                return Err(PlanError::InvalidWorkloadIdentity);
            }
            match tensor.role {
                TensorRole::Data | TensorRole::Auxiliary if tensor.dtype == DType::F32 => {}
                TensorRole::Index if matches!(tensor.dtype, DType::I32 | DType::I64 | DType::U64) => {
                    if tensor.accuracy != AccuracyContract::Exact || abs != 0.0 || rel != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
                TensorRole::IndexResult if tensor.dtype == DType::U64 => {
                    if tensor.accuracy != AccuracyContract::Exact || abs != 0.0 || rel != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
                TensorRole::Mask if tensor.dtype == DType::U8 => {
                    if tensor.accuracy != AccuracyContract::Exact || abs != 0.0 || rel != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
                _ => return Err(PlanError::InvalidWorkloadIdentity),
            }
            // This tensor_light variant is intentionally non-quantized. A
            // `None` quantization descriptor must also be canonical/empty.
            let q = tensor.quantization;
            if q.kind != QuantizationKind::None
                || q.granularity != ScaleGranularity::None
                || q.axis0 != 0
                || q.axis1 != 0
                || q.block0 != 0
                || q.block1 != 0
                || q.scale_tensor.is_some()
                || q.secondary_scale_tensor.is_some()
                || q.zero_point_tensor.is_some()
            {
                return Err(PlanError::InvalidQuantizationSpec);
            }

            // Use exactly the same dense-only predicate when validating an
            // offline graph as when validating its materialized frozen plan.
            tensor.sparsity.validate_dense_only()?;
            i += 1;
        }

        i = 0;
        while i < self.graph.nodes.len() {
            let node = &self.graph.nodes[i];
            // Node IDs follow canonical topological order. Independent nodes
            // therefore have one stable representation instead of arbitrary
            // user-assigned labels influencing the workload digest.
            if node.id != i as NodeId {
                return Err(PlanError::InvalidCanonicalGraph);
            }
            node.operation.validate_basic()?;
            // Canonical graph admission must agree with BOTH the semantic IO
            // validator and the executable CPU provider. F32, shape-preserving
            // Softmax/LogSoftmax and Normalization now have explicit built-in
            // CPU kernels, so they must not be rejected at this upper layer.
            // Unsupported GPU placements/artifacts are still rejected by the
            // frozen-plan/bind validators; admitting the semantic graph does
            // NOT advertise a GPU implementation or turn on runtime fallback.
            // Unimplemented legacy opcodes remain outside this release scope.
            if matches!(node.operation,
                Operation::Cast
                | Operation::FourierTransform(_)
                | Operation::DenseLinearAlgebra(_)
                | Operation::Convolution(_)
                | Operation::Attention(_)
                | Operation::Collective(_)
                | Operation::Pooling(_)
                | Operation::Custom { .. }
            ) {
                return Err(PlanError::InvalidOperationSpec);
            }
            if node.outputs.is_empty() {
                return Err(PlanError::InvalidCanonicalGraph);
            }
            let mut d = 0usize;
            while d < node.dependencies.len() {
                let dep = node.dependencies[d];
                if d > 0 && node.dependencies[d - 1] >= dep {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                let mut dep_pos = None;
                let mut p = 0usize;
                while p < i {
                    if self.graph.nodes[p].id == dep {
                        dep_pos = Some(p);
                        break;
                    }
                    p += 1;
                }
                if dep_pos.is_none() {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                d += 1;
            }

            let mut v = 0usize;
            while v < node.inputs.len() {
                if self.find_tensor(node.inputs[v]).is_none() {
                    return Err(PlanError::MissingTensor);
                }
                v += 1;
            }
            v = 0;
            while v < node.outputs.len() {
                let output = node.outputs[v];
                if self.find_tensor(output).is_none() {
                    return Err(PlanError::MissingTensor);
                }
                // Canonical dataflow is logical SSA. In-place execution is a
                // physical storage/alias decision in the FrozenExecutionPlan,
                // never a logical overwrite of the same TensorId.
                if node.inputs.contains(&output) {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                let mut same_node = v + 1;
                while same_node < node.outputs.len() {
                    if node.outputs[same_node] == output {
                        return Err(PlanError::DuplicateTensorProducer);
                    }
                    same_node += 1;
                }
                let mut previous = 0usize;
                while previous < i {
                    if self.graph.nodes[previous].outputs.contains(&output) {
                        return Err(PlanError::DuplicateTensorProducer);
                    }
                    previous += 1;
                }
                v += 1;
            }

            // Reject semantically invalid static operations before candidate
            // generation or benchmarking. The same validator is reused after
            // mapping/fusion so offline and frozen-plan semantics cannot drift.
            super::validation::validate_canonical_operation_io(
                self.graph.tensors,
                node.operation,
                node.inputs,
                node.outputs,
            )?;

            // Every data dependency is explicit and topologically valid. A
            // tensor produced by a later node is not an external graph input;
            // accepting it here would make the canonical order non-executable.
            v = 0;
            while v < node.inputs.len() {
                let tensor = node.inputs[v];
                let mut producer_position = None;
                let mut p = 0usize;
                while p < self.graph.nodes.len() {
                    if self.graph.nodes[p].outputs.contains(&tensor) {
                        if producer_position.is_some() {
                            return Err(PlanError::DuplicateTensorProducer);
                        }
                        producer_position = Some(p);
                    }
                    p += 1;
                }
                if let Some(position) = producer_position {
                    if position >= i {
                        return Err(PlanError::InvalidDependencyOrder);
                    }
                    let producer = self.graph.nodes[position].id;
                    if !node.dependencies.contains(&producer) {
                        return Err(PlanError::MissingDataDependency);
                    }
                }
                v += 1;
            }

            // Canonical dependencies are exactly the direct data producers.
            // Extra control-only edges belong in the frozen schedule/sync plan;
            // allowing them here would create multiple canonical encodings of
            // the same dataflow workload.
            d = 0;
            while d < node.dependencies.len() {
                let dependency = node.dependencies[d];
                let producer = self
                    .find_node(dependency)
                    .ok_or(PlanError::InvalidCanonicalGraph)?;
                let mut feeds_input = false;
                let mut input_index = 0usize;
                while input_index < node.inputs.len() {
                    if producer.outputs.contains(&node.inputs[input_index]) {
                        feeds_input = true;
                        break;
                    }
                    input_index += 1;
                }
                if !feeds_input {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                d += 1;
            }
            i += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod normalized_cpu_graph_admission_tests {
    use super::*;

    // Exercise the actual canonical graph checker (not only an operation
    // enum whitelist). Runtime hardware/binding admission has separate tests.
    fn f32_tensor(id: TensorId, shape: StaticShape) -> TensorSpec {
        let mut strides = [0isize; MAX_RANK];
        let mut stride = 1isize;
        for axis in (0..shape.rank()).rev() {
            strides[axis] = stride;
            let dim = isize::try_from(shape.dim(axis).expect("valid axis")).expect("small test shape");
            stride = stride.checked_mul(dim).expect("small test shape");
        }
        TensorSpec {
            id,
            shape,
            dtype: DType::F32,
            role: TensorRole::Data,
            layout: TensorLayout {
                kind: LayoutKind::RowMajorContiguous,
                strides,
                block: [0; MAX_RANK],
                alignment_bytes: 4,
            },
            accuracy: AccuracyContract::StrictFp32,
            tolerance: NumericalTolerance {
                max_abs_error_f32_bits: 0,
                max_rel_error_f32_bits: 0,
            },
            reuse: ReusePattern::SingleUse,
            sparsity: SparsitySpec {
                pattern: SparsityPattern::Dense,
                block_m: 0,
                block_n: 0,
                metadata_tensor: None,
            },
            quantization: QuantizationSpec {
                kind: QuantizationKind::None,
                granularity: ScaleGranularity::None,
                axis0: 0,
                axis1: 0,
                block0: 0,
                block1: 0,
                scale_tensor: None,
                secondary_scale_tensor: None,
                zero_point_tensor: None,
            },
        }
    }

    fn check_canonical_graph(
        operation: Operation, input_shape: StaticShape, output_shape: StaticShape,
    ) -> Result<(), PlanError> {
        // The canonical IR uses static slices. Leak tiny fixtures in tests;
        // nothing here is part of a production runtime or performance path.
        let tensors: &'static [TensorSpec] = Box::leak(Box::new([
            f32_tensor(0, input_shape),
            f32_tensor(1, output_shape),
        ]));
        let nodes: &'static [CanonicalNode] = Box::leak(Box::new([
            CanonicalNode {
                id: 0,
                operation,
                dependencies: &[],
                inputs: &[0],
                outputs: &[1],
            },
        ]));
        let graph = CanonicalWorkload {
            graph: CanonicalStaticDataflow { tensors, nodes },
            hardware: super::super::target::HardwareContract::canonical(None, &[]),
            memory_topology: MemoryTopologyContract {
                host_numa_nodes: &[],
                host_numa_distances: &[],
                gpu_device_ordinals: &[],
                peer_access: &[],
                supports_pinned_host_memory: false,
                supports_mapped_host_memory: false,
                supports_unified_memory: false,
            },
            execution_topology: ExecutionTopologyContract {
                cpu_enabled: false,
                gpu_device_ordinals: &[],
                allow_cpu_gpu_partitioned: false,
                allow_cpu_gpu_pipeline: false,
            },
        };
        // Isolate semantic graph validation from hardware placement. A real
        // FrozenExecutionPlan also needs a supported CPU artifact and ABI.
        graph.validate_graph()
    }

    #[test]
    fn softmax_and_all_six_f32_normalizations_reach_canonical_graph() {
        let shape = StaticShape::Rank3([2, 4, 3]);
        for log in [false, true] {
            assert_eq!(check_canonical_graph(
                Operation::Softmax { axis: 2, log }, shape, shape,
            ), Ok(()));
        }
        let epsilon_f32_bits = 1e-5_f32.to_bits();
        let cases = [
            (NormalizationKind::LayerNorm, 0b100, 0, 0),
            (NormalizationKind::RmsNorm, 0b100, 0, 0),
            (NormalizationKind::L2Norm, 0b100, 0, 0),
            (NormalizationKind::BatchNorm, 0b101, 1, 0),
            (NormalizationKind::InstanceNorm, 0b100, 1, 0),
            (NormalizationKind::GroupNorm, 0b110, 1, 2),
        ];
        for (kind, axes, channel_axis, groups) in cases {
            let operation = Operation::Normalization(NormalizationSpec {
                kind,
                axes: AxisMask(axes),
                epsilon_f32_bits,
                channel_axis,
                groups,
            });
            assert_eq!(check_canonical_graph(operation, shape, shape), Ok(()),
                "should admit semantic f32 graph: {kind:?}");
        }
    }

    #[test]
    fn high_rank_axes_and_empty_batch_do_not_change_f32_shape_contract() {
        let shape = StaticShape::Rank8([1, 2, 1, 1, 1, 1, 1, 3]);
        assert_eq!(check_canonical_graph(
            Operation::Softmax { axis: 7, log: false }, shape, shape,
        ), Ok(()));
        assert_eq!(check_canonical_graph(
            Operation::Normalization(NormalizationSpec {
                kind: NormalizationKind::LayerNorm,
                axes: AxisMask(0b1000_0000),
                epsilon_f32_bits: 1e-5_f32.to_bits(),
                groups: 0,
                channel_axis: 0,
            }), shape, shape,
        ), Ok(()));
        assert_eq!(check_canonical_graph(
            Operation::Normalization(NormalizationSpec {
                kind: NormalizationKind::GroupNorm,
                axes: AxisMask(0b1111_1110),
                epsilon_f32_bits: 1e-5_f32.to_bits(),
                groups: 2,
                channel_axis: 1,
            }), shape, shape,
        ), Ok(()));
        let empty_batch = StaticShape::Rank3([0, 4, 3]);
        assert_eq!(check_canonical_graph(
            Operation::Softmax { axis: 2, log: true }, empty_batch, empty_batch,
        ), Ok(()));
        assert_eq!(check_canonical_graph(
            Operation::Normalization(NormalizationSpec {
                kind: NormalizationKind::GroupNorm,
                axes: AxisMask(0b110),
                epsilon_f32_bits: 1e-5_f32.to_bits(),
                groups: 2,
                channel_axis: 1,
            }), empty_batch, empty_batch,
        ), Ok(()));
    }

    #[test]
    fn admitted_f32_ops_still_reject_invalid_axes_or_changed_shape() {
        let shape = StaticShape::Rank3([2, 4, 3]);
        assert_eq!(check_canonical_graph(
            Operation::Softmax { axis: 3, log: false }, shape, shape,
        ), Err(PlanError::InvalidOperationSpec));
        assert_eq!(check_canonical_graph(
            Operation::Softmax { axis: 2, log: true },
            shape, StaticShape::Rank3([2, 4, 2]),
        ), Err(PlanError::InvalidOperationSpec));
        for (kind, axes, channel_axis, groups) in [
            (NormalizationKind::BatchNorm, 0b100, 1, 0),
            (NormalizationKind::InstanceNorm, 0b101, 1, 0),
            (NormalizationKind::GroupNorm, 0b110, 1, 3),
        ] {
            assert_eq!(check_canonical_graph(
                Operation::Normalization(NormalizationSpec {
                    kind,
                    axes: AxisMask(axes),
                    epsilon_f32_bits: 1e-5_f32.to_bits(),
                    groups,
                    channel_axis,
                }), shape, shape,
            ), Err(PlanError::InvalidOperationSpec));
        }
        assert_eq!(check_canonical_graph(
            Operation::Normalization(NormalizationSpec {
                kind: NormalizationKind::LayerNorm,
                axes: AxisMask(0b100),
                epsilon_f32_bits: 1e-5_f32.to_bits(),
                groups: 0,
                channel_axis: 0,
            }), shape, StaticShape::Rank3([2, 4, 2]),
        ), Err(PlanError::InvalidOperationSpec));
        // Scope reduction is intentional: unsafe Cast remains rejected.
        assert_eq!(check_canonical_graph(
            Operation::Cast, shape, shape,
        ), Err(PlanError::InvalidOperationSpec));
    }
}
