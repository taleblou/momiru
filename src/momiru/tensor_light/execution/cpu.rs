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
// Subcomponent:  CPU
// Module:        momiru::tensor_light::execution::cpu
// Source:        src/tensor_light/execution/cpu.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Built-in restricted f32 CPU provider for prevalidated UASDEA frozen-plan
//   execution.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Bind validated CPU numeric artifacts and hardware feature requirements.
//   - Resolve shape, pointer, numeric-mode and kernel decisions before replay.
//   - Execute prebound CPU work without runtime search or hidden allocation.
//   - Avoid creating a second eager tensor arithmetic API.
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

//! Restricted, built-in CPU FP32 provider for the existing UASDEA frozen-plan
//! runtime. This is not a second eager tensor arithmetic API or a GPU shim.
//!
//! All shapes, pointers, numeric modes, hardware feature gates and function
//! pointers are resolved at BIND. REPLAY calls an already-selected function
//! with no allocation, dispatch search, new threads or plan inspection.
//!
//! Only exactly named, linked-in CPU functions are accepted. This module is
//! NOT a signature verifier or signed artifact loader. External native images,
//! vendor libraries and compiler-generated kernels require a separate trusted
//! catalog and independently executed attestation before release.

use std::sync::OnceLock;
use std::ptr;

use super::{
    ArtifactKind, Backend, BindingAccess, BoundBackend, BufferHandle,
    CpuDispatchMode, CpuKernelPlan, CpuParallelKind, CpuKernelSource, RustFloatMode,
    RustCodegenPlan, RustPanicStrategy, LtoMode, PgoMode, PostLinkOptimization,
    PrefetchKind, StorePolicy, NumaPolicy, WorkerWaitPolicy, ElementwiseKind, ScalarElementwiseKind, ReductionKind, AxisSelector, ScanKind, ScanOutput, SortOutput, SortNanPolicy, NormalizationKind, AxisMask, AxisList,
    FrozenExecutionPlan, FrozenNode, KernelFamily, KernelMathMode,
    LayoutKind, MemoryClass, HostMemoryPolicy, DeviceAllocationPolicy,
    MemoryArenaPlan, PagePolicy, Operation, Placement, PlanError,
    RuntimeCapabilityBinding, ScheduleItem, StaticShape, StorageAccess, WorkRange,
    StorageBinding, TensorSpec, cpu_feature, FeatureMask,
};
use crate::tensor_light::tensor::DType;

/// A linked-in source entry name, NOT an external native-image signature.
pub const BUILTIN_CPU_ARTIFACT: &str = "momiru::cpu::built_in_f32";
pub const BUILTIN_COPY: &str = "copy_contiguous_f32";
pub const BUILTIN_ADD: &str = "add_contiguous_f32";
pub const BUILTIN_SUB: &str = "sub_contiguous_f32";
pub const BUILTIN_MUL: &str = "mul_contiguous_f32";
pub const BUILTIN_DIV: &str = "div_contiguous_f32";
pub const BUILTIN_DIV_AVX2: &str = "div_contiguous_avx2_f32";
pub const BUILTIN_DIV_ZERO_TO_ZERO: &str = "divide_zero_to_zero_contiguous_f32";
pub const BUILTIN_RECIPROCAL: &str = "reciprocal_contiguous_f32";
pub const BUILTIN_RECIPROCAL_ZERO_TO_ZERO: &str = "reciprocal_zero_to_zero_contiguous_f32";
pub const BUILTIN_SQRT_NONPOSITIVE_TO_ZERO: &str = "sqrt_nonpositive_to_zero_contiguous_f32";
pub const BUILTIN_GEMM: &str = "matmul_contiguous_f32";
/// Each distinct native implementation has its own exact artifact entry.
pub const BUILTIN_GEMM_AXPY: &str = "matmul_contiguous_axpy_f32";
pub const BUILTIN_GEMM_AVX2: &str = "matmul_contiguous_avx2_f32";
pub const BUILTIN_ADD_AVX2: &str = "add_contiguous_avx2_f32";
pub const BUILTIN_SUB_AVX2: &str = "sub_contiguous_avx2_f32";
pub const BUILTIN_MUL_AVX2: &str = "mul_contiguous_avx2_f32";
pub const BUILTIN_NEG: &str = "neg_contiguous_f32";
pub const BUILTIN_ABS: &str = "abs_contiguous_f32";
pub const BUILTIN_SQUARE: &str = "square_contiguous_f32";
pub const BUILTIN_SQRT: &str = "sqrt_contiguous_f32";
pub const BUILTIN_MIN: &str = "minimum_contiguous_f32";
pub const BUILTIN_MAX: &str = "maximum_contiguous_f32";
pub const BUILTIN_SUM: &str = "sum_all_contiguous_f32";
pub const BUILTIN_PRODUCT: &str = "product_all_contiguous_f32";
pub const BUILTIN_COUNT_NONZERO: &str = "count_nonzero_all_contiguous_f32";
pub const BUILTIN_REDUCE_MIN: &str = "minimum_all_contiguous_f32";
pub const BUILTIN_REDUCE_MAX: &str = "maximum_all_contiguous_f32";
pub const BUILTIN_REDUCE_MEAN: &str = "mean_all_contiguous_f32";
pub const BUILTIN_REDUCE_L1: &str = "l1_norm_all_contiguous_f32";
pub const BUILTIN_REDUCE_L2: &str = "l2_norm_all_contiguous_f32";
pub const BUILTIN_REDUCE_LINF: &str = "linf_norm_all_contiguous_f32";
pub const BUILTIN_REDUCE_VARIANCE: &str = "variance_population_all_contiguous_f32";
pub const BUILTIN_REDUCE_STDDEV: &str = "stddev_population_all_contiguous_f32";
pub const BUILTIN_REDUCE_LOGSUMEXP: &str = "logsumexp_all_contiguous_f32";
// Frozen scalar-immediate kernels never interpret a pointer as an extra dtype.
pub const BUILTIN_SCALAR_ADD: &str = "scalar_add_contiguous_f32";
pub const BUILTIN_SCALAR_MUL: &str = "scalar_multiply_contiguous_f32";
pub const BUILTIN_SCALAR_DIV_ZERO: &str = "scalar_divide_zero_to_zero_contiguous_f32";
pub const BUILTIN_SCALAR_LEAKY_RELU: &str = "scalar_leaky_relu_contiguous_f32";
pub const BUILTIN_SCALAR_CLAMP: &str = "scalar_clamp_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_IDENTITY: &str = "scalar_power_identity_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_SQUARE: &str = "scalar_power_square_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_SAFE_SQRT: &str = "scalar_power_safe_sqrt_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_SAFE_RECIP: &str = "scalar_power_safe_recip_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_3_2: &str = "scalar_power_three_halves_contiguous_f32";
pub const BUILTIN_SCALAR_POWER_GENERAL: &str = "scalar_power_general_contiguous_f32";

// Additional exact, contiguous f32 operations; each has a distinct frozen
// artifact entry so a different numeric contract cannot be substituted.
pub const BUILTIN_HYPOT: &str = "hypot_contiguous_f32";
pub const BUILTIN_PRELU: &str = "prelu_contiguous_f32";
pub const BUILTIN_RSQRT: &str = "rsqrt_contiguous_f32";
pub const BUILTIN_RSQRT_NONPOSITIVE_TO_ZERO: &str = "rsqrt_nonpositive_to_zero_contiguous_f32";
pub const BUILTIN_EXP: &str = "exp_contiguous_f32";
pub const BUILTIN_EXPM1: &str = "expm1_contiguous_f32";
pub const BUILTIN_LOG: &str = "log_contiguous_f32";
pub const BUILTIN_LOG_NONPOSITIVE_TO_NEG87: &str = "log_nonpositive_to_neg87_contiguous_f32";
pub const BUILTIN_LOG1P: &str = "log1p_contiguous_f32";
pub const BUILTIN_LOG1P_BELOW_MINUS_ONE_TO_NEG87: &str = "log1p_below_minus_one_to_neg87_contiguous_f32";
pub const BUILTIN_LOG1P_ABS: &str = "log1p_abs_contiguous_f32";
pub const BUILTIN_SIN: &str = "sin_contiguous_f32";
pub const BUILTIN_COS: &str = "cos_contiguous_f32";
pub const BUILTIN_TAN: &str = "tan_contiguous_f32";
pub const BUILTIN_ERF: &str = "erf_contiguous_f32";
pub const BUILTIN_GELU_ERF: &str = "gelu_erf_contiguous_f32";
pub const BUILTIN_TANH: &str = "tanh_contiguous_f32";
// Scalar-only activations with individually named AOT entrypoints. No
// approximate/fast-math kernel may be substituted for these functions.
pub const BUILTIN_SIGMOID: &str = "sigmoid_contiguous_f32";
pub const BUILTIN_RELU: &str = "relu_contiguous_f32";
pub const BUILTIN_RELU6: &str = "relu6_contiguous_f32";
pub const BUILTIN_SILU: &str = "silu_contiguous_f32";
pub const BUILTIN_SOFTPLUS: &str = "softplus_contiguous_f32";
pub const BUILTIN_SOFTSIGN: &str = "softsign_contiguous_f32";
pub const BUILTIN_HARDSWISH: &str = "hardswish_contiguous_f32";
pub const BUILTIN_MISH: &str = "mish_contiguous_f32";
// Both activations have one fixed, portable f32 semantic identity. Neither
// requires an extra parameter tensor or changes the output rank/shape.
pub const BUILTIN_ELU: &str = "elu_alpha_one_contiguous_f32";
pub const BUILTIN_SELU: &str = "selu_canonical_constants_contiguous_f32";
pub const BUILTIN_FLOOR: &str = "floor_contiguous_f32";
pub const BUILTIN_CEIL: &str = "ceil_contiguous_f32";
pub const BUILTIN_ROUND: &str = "round_ties_away_contiguous_f32";
pub const BUILTIN_SIGN: &str = "sign_signed_zero_contiguous_f32";
pub const BUILTIN_GELU_TANH: &str = "gelu_tanh_approx_contiguous_f32";

// All of these operations keep their exact input rank, shape and f32 dtype.
// Offline binding freezes the normalized axis into (outer, axis, inner) extents.
// The replayed entry performs no plan inspection, allocation or dispatch.
pub const BUILTIN_SOFTMAX: &str = "softmax_contiguous_axis_f32";
pub const BUILTIN_LOG_SOFTMAX: &str = "log_softmax_contiguous_axis_f32";
// Reverse a single axis or a contiguous block of axes without changing
// tensor shape, dtype, rank or the raw IEEE-754 payload bytes.
pub const BUILTIN_FLIP_AXIS_BLOCK: &str = "flip_contiguous_axis_block_f32";
/// Arbitrary validated, nonconsecutive axis subsets use a distinct entry.
/// The contiguous/block entry remains the preferred faster implementation.
pub const BUILTIN_FLIP_GENERAL: &str = "flip_contiguous_arbitrary_axes_f32";
// Stable value sort preserves the source f32 payload bits, shape and rank.
// Original positions are tie-breakers so an allocation-free unstable sort of
// fixed (bit-pattern, ordinal) records reproduces stable input ordering.
pub const BUILTIN_SORT_ASC_NANS_FIRST: &str = "sort_values_ascending_nans_first_f32";
pub const BUILTIN_SORT_ASC_NANS_LAST: &str = "sort_values_ascending_nans_last_f32";
pub const BUILTIN_SORT_DESC_NANS_FIRST: &str = "sort_values_descending_nans_first_f32";
pub const BUILTIN_SORT_DESC_NANS_LAST: &str = "sort_values_descending_nans_last_f32";
pub const BUILTIN_SCAN_SUM_INCLUSIVE: &str = "scan_sum_inclusive_contiguous_axis_f32";
pub const BUILTIN_SCAN_SUM_EXCLUSIVE: &str = "scan_sum_exclusive_contiguous_axis_f32";
pub const BUILTIN_SCAN_PRODUCT: &str = "scan_product_inclusive_contiguous_axis_f32";
pub const BUILTIN_SCAN_MINIMUM: &str = "scan_minimum_inclusive_contiguous_axis_f32";
pub const BUILTIN_SCAN_MAXIMUM: &str = "scan_maximum_inclusive_contiguous_axis_f32";
pub const BUILTIN_SCAN_LOGSUMEXP: &str = "scan_logsumexp_inclusive_contiguous_axis_f32";
// F32-only normalization over a contiguous, nonempty suffix of logical axes.
// The same input shape and dtype is preserved; CPU f64 scalar accumulation
// avoids overflow without constructing or exporting a different Tensor dtype.
pub const BUILTIN_LAYERNORM: &str = "layernorm_contiguous_suffix_f32";
pub const BUILTIN_RMSNORM: &str = "rmsnorm_contiguous_suffix_f32";
pub const BUILTIN_L2_NORMALIZATION: &str = "l2_normalize_contiguous_suffix_f32";
/// Fallback for valid, non-suffix normalization axes. No intermediate tensor
/// allocation and no change to the public f32 tensor's shape or rank.
pub const BUILTIN_LAYERNORM_AXES: &str = "layernorm_contiguous_arbitrary_axes_f32";
pub const BUILTIN_RMSNORM_AXES: &str = "rmsnorm_contiguous_arbitrary_axes_f32";
pub const BUILTIN_L2_NORMALIZATION_AXES: &str = "l2_normalize_contiguous_arbitrary_axes_f32";
// BatchNorm and InstanceNorm use the IR's population statistics from the
// current input (no implicit running mean/variance or affine parameters).
// Their distinct frozen entry labels prevent accidental cross-operation bind.
pub const BUILTIN_BATCH_NORM_AXES: &str = "batch_norm_population_arbitrary_axes_f32";
pub const BUILTIN_INSTANCE_NORM_SUFFIX: &str = "instance_norm_population_suffix_f32";
pub const BUILTIN_INSTANCE_NORM_AXES: &str = "instance_norm_population_arbitrary_axes_f32";
// A GroupNorm with channel axis 1 (or one group) is physically contiguous
// per group and reuses the direct f32 LayerNorm-style population kernel.
pub const BUILTIN_GROUP_NORM_CONTIGUOUS: &str = "group_norm_population_contiguous_groups_f32";
pub const BUILTIN_GROUP_NORM_CHANNEL_AXIS: &str = "group_norm_population_arbitrary_channel_axis_f32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuNativeError {
    NotConfigured,
    AlreadyConfigured,
    InvalidCapabilities,
    InvalidPlan(PlanError),
    UnsupportedDevice,
    UnsupportedMemory,
    UnsupportedOperation,
    UnsupportedLayout,
    UnsupportedMathMode,
    UnsupportedArtifact,
    UnsupportedCpuFeature,
    UnsafeAliasing,
    MissingBinding,
    InvalidShape,
    InvalidNode,
}

impl core::fmt::Display for CpuNativeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CpuNativeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CpuInstallation {
    capabilities: RuntimeCapabilityBinding,
    artifact_hash_hi: u64,
    artifact_hash_lo: u64,
}

static INSTALLED_CPU: OnceLock<CpuInstallation> = OnceLock::new();

/// Install the exact, already-detected CPU-only capability identity and the
/// deployed *trusted built-in image* identifier once, prior to preparing plans.
/// A provider of a verifiable release image must supply that image's digest;
/// these two 64-bit fields merely bind a plan to the caller's trusted image
/// identity, and do NOT independently authenticate the executable or signature.
/// No hardware heuristics, tuners or CPU/GPU routing run during replay.
pub fn configure_builtin_cpu(
    capabilities: RuntimeCapabilityBinding,
    artifact_hash_hi: u64,
    artifact_hash_lo: u64,
) -> Result<(), CpuNativeError> {
    // A CPU-only execution PLAN is valid on a laptop with an installed GPU.
    // GPU hardware may appear in the exact capability fingerprint; GPU nodes,
    // device storage and transfers remain rejected at native resource bind.
    if !capabilities.is_valid()
        || capabilities.hardware.cpu.is_none()
        || (artifact_hash_hi | artifact_hash_lo) == 0
        || capabilities.hardware.validate_canonical().is_err()
        || capabilities.memory_topology.canonical_hash()
            != (capabilities.memory_topology_hash_hi, capabilities.memory_topology_hash_lo)
    {
        return Err(CpuNativeError::InvalidCapabilities);
    }
    let install = CpuInstallation { capabilities, artifact_hash_hi, artifact_hash_lo };
    if let Some(previous) = INSTALLED_CPU.get() {
        return if *previous == install { Ok(()) } else { Err(CpuNativeError::AlreadyConfigured) };
    }
    match INSTALLED_CPU.set(install) {
        Ok(()) => Ok(()),
        Err(_) if INSTALLED_CPU.get() == Some(&install) => Ok(()),
        Err(_) => Err(CpuNativeError::AlreadyConfigured),
    }
}

/// CPU backend integrates with the one existing `ExecutionBinder` and its
/// retained external-storage ownership leases.
pub struct CpuNativeBackend;

/// Full physical tensor-storage region, including padding outside its
/// contiguous logical payload. `noalias_io` promises that these regions,
/// not merely the elements touched by a kernel, do not overlap.
#[derive(Clone, Copy)]
struct PhysicalRange {
    start: usize,
    bytes: usize,
}

impl PhysicalRange {
    #[inline]
    fn disjoint_from(self, other: Self) -> bool {
        if self.bytes == 0 || other.bytes == 0 { return true; }
        let a = self.start as u128;
        let b = other.start as u128;
        a + self.bytes as u128 <= b || b + other.bytes as u128 <= a
    }
}

#[derive(Clone, Copy)]
struct Span {
    ptr: *mut f32,
    len: usize,
    shape: StaticShape,
    physical: PhysicalRange,
}

impl Span {
    #[inline]
    fn disjoint_from(self, other: Self) -> bool {
        if self.len == 0 || other.len == 0 { return true; }
        let a = self.ptr as usize as u128;
        let b = other.ptr as usize as u128;
        let a_end = a + self.len as u128 * 4;
        let b_end = b + other.len as u128 * 4;
        a_end <= b || b_end <= a
    }
}

/// Bind-verified, small-rank metadata used ONLY by slower generic-axis
/// fallbacks. Fast contiguous kernels never inspect it during replay.
#[derive(Clone, Copy)]
struct ArbitraryAxes {
    dims: [usize; 8],
    rank: u8,
    mask: u8,
}

/// Fixed-size sorting record, allocated during BIND, not while replaying.
/// Bits are copied unchanged (including distinct NaNs and signed zeros).
#[derive(Clone, Copy)]
struct SortRecord {
    bits: u32,
    ordinal: usize,
}

#[derive(Clone, Copy)]
struct CpuCommand {
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    lhs_vector: bool,
    rhs_vector: bool,
    scalar0: f32,
    scalar1: f32,
    arbitrary_axes: Option<ArbitraryAxes>,
    sort_scratch: *mut SortRecord,
    entry: unsafe fn(&CpuCommand),
}

/// Prepared nodes are indexed by frozen node ID. No runtime map/hash/probing.
pub struct BoundCpuNative {
    commands: Vec<Option<CpuCommand>>,
    // A single scratch allocation serves every sorting node, since this
    // backend executes nodes synchronously under an exclusive &mut self.
    // The buffer is allocated at BIND and never resized during replay.
    _sort_scratch: Vec<SortRecord>,
}

unsafe impl BoundBackend for BoundCpuNative {
    type Error = CpuNativeError;

    #[inline]
    fn execute_cpu_node(
        &mut self, node: &'static FrozenNode, _kernel: &'static CpuKernelPlan,
    ) -> Result<(), Self::Error> {
        let command = self.commands.get(node.id as usize)
            .and_then(Option::as_ref)
            .ok_or(CpuNativeError::InvalidNode)?;
        // SAFETY: `CpuNativeBackend::bind_resources` checked all source/output
        // spans, strict layout, exact type/shape and no output/input aliasing.
        // `BoundExecution` owns the lifetime guards. No async work is launched.
        unsafe { (command.entry)(command) };
        Ok(())
    }

    fn execute_gpu_node(&mut self, _: &'static FrozenNode, _: &'static super::GpuKernelPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn launch_cpu_slice(&mut self, _: &'static FrozenNode, _: u16, _: &'static super::CpuGpuSlice, _: &'static CpuKernelPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn launch_gpu_slice(&mut self, _: &'static FrozenNode, _: u16, _: &'static super::CpuGpuSlice, _: &'static super::GpuKernelPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn complete_cpu_gpu_slice(&mut self, _: &'static FrozenNode, _: u16, _: &'static super::CpuGpuSlice) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn execute_gpu_graph_segment(&mut self, _: &'static super::GpuGraphSegmentPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn execute_transfer(&mut self, _: &'static super::TransferPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn execute_host_readback(&mut self, _: &'static super::TransferPlan, _: super::HostBoundaryClass) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn execute_transfer_batch(&mut self, _: &'static super::TransferBatchPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    fn execute_sync(&mut self, _: &'static super::SyncPlan) -> Result<(), Self::Error> {
        Err(CpuNativeError::UnsupportedDevice)
    }
    /// All built-in work is synchronous. There are no outstanding device
    /// events, detached workers or deferred reads when the command returns.
    fn quiesce_or_abort(&mut self) {}
}

unsafe impl Backend for CpuNativeBackend {
    type Bound = BoundCpuNative;
    type Error = CpuNativeError;

    fn bind_capabilities() -> Result<RuntimeCapabilityBinding, Self::Error> {
        INSTALLED_CPU.get().map(|item| item.capabilities)
            .ok_or(CpuNativeError::NotConfigured)
    }

    unsafe fn bind_resources(
        plan: &'static FrozenExecutionPlan,
        capabilities: RuntimeCapabilityBinding,
        bindings: &[StorageBinding<'_>],
    ) -> Result<Self::Bound, Self::Error> {
        let installed = INSTALLED_CPU.get().ok_or(CpuNativeError::NotConfigured)?;
        if installed.capabilities != capabilities || !capabilities.matches_plan(plan) {
            return Err(CpuNativeError::InvalidCapabilities);
        }
        // Structural validation is once-only in PlanDb::from_static_plans.
        // Physical binding validation is once per bind in PreparedExecution.
        // This is an unsafe entrypoint: direct callers MUST establish both
        // preconditions (documented on Backend::bind_resources) themselves.
        // Repeating complete graph checks here would penalize every warm bind.
        if !plan.execution_contexts.is_empty() || !plan.streams.is_empty()
            || !plan.events.is_empty() || !plan.gpu_graph_segments.is_empty()
            || !plan.transfers.is_empty() || !plan.transfer_batches.is_empty()
            || !plan.synchronization.is_empty()
        {
            return Err(CpuNativeError::UnsupportedDevice);
        }
        if plan.arenas.iter().any(|arena| !builtin_host_arena_admissible(arena))
            || plan.storages.iter().any(|storage| {
                storage.residency != super::ResidencyPolicy::ExternalBound
                    || storage.external_alias_of.is_some()
            })
        {
            return Err(CpuNativeError::UnsupportedMemory);
        }
        let mut commands = Vec::new();
        commands.try_reserve_exact(plan.nodes.len())
            .map_err(|_| CpuNativeError::UnsupportedMemory)?;
        commands.resize(plan.nodes.len(), None);
        let mut maximum_sort_axis = 0usize;
        for node in plan.nodes {
            if node.placement != Placement::Cpu || !node.fused_operations.is_empty() {
                return Err(CpuNativeError::UnsupportedDevice);
            }
            if node.id as usize >= commands.len() || commands[node.id as usize].is_some() {
                return Err(CpuNativeError::InvalidNode);
            }
            let kernel = node.cpu.ok_or(CpuNativeError::UnsupportedOperation)?;
            let command = prepare_node(plan, node, kernel, bindings, *installed)?;
            if matches!(node.operation, Operation::Sort(_)) && command.len != 0 {
                // Geometry has proved k <= len, so this bound is finite and
                // also covers every slice of every other sort node.
                maximum_sort_axis = maximum_sort_axis.max(command.k);
            }
            commands[node.id as usize] = Some(command);
        }
        // No metadata-only graph node or unsupported schedule item can be
        // silently skipped by treating it as a successfully executed kernel.
        for item in plan.schedule {
            match item {
                ScheduleItem::Node(id) => {
                    if commands.get(*id as usize).and_then(Option::as_ref).is_none() {
                        return Err(CpuNativeError::InvalidNode);
                    }
                }
                _ => return Err(CpuNativeError::UnsupportedOperation),
            }
        }
        // Schedule admission above guarantees the nodes will run serially
        // through &mut BoundCpuNative. Allocate ONLY the largest sort-axis
        // buffer (not the sum across sorting nodes), and point each command
        // at the same retained heap allocation. The Vec is never resized or
        // replaced once these raw pointers are installed.
        let mut sort_scratch: Vec<SortRecord> = Vec::new();
        if maximum_sort_axis != 0 {
            sort_scratch.try_reserve_exact(maximum_sort_axis)
                .map_err(|_| CpuNativeError::UnsupportedMemory)?;
            sort_scratch.resize(maximum_sort_axis, SortRecord { bits: 0, ordinal: 0 });
            let shared = sort_scratch.as_mut_ptr();
            for node in plan.nodes {
                if matches!(node.operation, Operation::Sort(_)) {
                    let command = commands.get_mut(node.id as usize)
                        .and_then(Option::as_mut)
                        .ok_or(CpuNativeError::InvalidNode)?;
                    if command.len != 0 { command.sort_scratch = shared; }
                }
            }
        }
        Ok(BoundCpuNative { commands, _sort_scratch: sort_scratch })
    }
}

/// The basic linked-in CPU backend receives an externally owned, ordinary
/// host allocation. It cannot verify or arrange NUMA placement, prefaulting,
/// transparent/explicit huge pages, or any device/memory-pool contract. Such
/// configurations are valid in the generic schema, but must not silently
/// bind to this implementation which merely receives a host pointer.
#[inline]
fn builtin_host_arena_admissible(arena: &MemoryArenaPlan) -> bool {
    arena.class == MemoryClass::HostAligned
        && arena.host_policy == HostMemoryPolicy::Pageable
        && arena.device_allocation == DeviceAllocationPolicy::NotApplicable
        && arena.page_policy == PagePolicy::BasePages
        && arena.page_size_bytes == 0
        && !arena.prefault
        && arena.numa_node.is_none()
        && arena.numa_interleave_nodes.is_empty()
        && arena.device_ordinal.is_none()
        && arena.locality_domain.is_none()
        && arena.multicast_devices.is_empty()
        && arena.memory_pool.is_none()
        && arena.unified_advice.is_none()
}

fn resolve_span(
    plan: &FrozenExecutionPlan, storage_id: u32, bindings: &[StorageBinding<'_>],
) -> Result<Span, CpuNativeError> {
    let storage = plan.storages.get(storage_id as usize)
        .filter(|item| item.id == storage_id)
        .ok_or(CpuNativeError::MissingBinding)?;
    let tensor: &TensorSpec = plan.tensors.iter().find(|item| item.id == storage.tensor)
        .ok_or(CpuNativeError::InvalidShape)?;
    if tensor.dtype != DType::F32
        || storage.physical_layout.kind != LayoutKind::RowMajorContiguous
        || !storage.physical_layout.is_row_major_contiguous_for(&tensor.shape)
    {
        return Err(CpuNativeError::UnsupportedLayout);
    }
    let count = tensor.shape.element_count().ok_or(CpuNativeError::InvalidShape)?;
    let bytes = tensor.shape.byte_count(DType::F32).ok_or(CpuNativeError::InvalidShape)?;
    // A contiguous tensor can occupy an offset subregion of its own physical
    // storage. Preserve the frozen base-offset and reject anything not fully
    // covered by the certified physical allocation (including arithmetic
    // overflow). A padded external arena does not change f32 or tensor rank.
    if !contiguous_payload_fits(storage.physical_bytes, storage.base_offset_bytes, bytes) {
        return Err(CpuNativeError::UnsupportedLayout);
    }
    let slot = plan.memory_slots.get(storage.slot as usize)
        .filter(|item| item.id == storage.slot)
        .ok_or(CpuNativeError::MissingBinding)?;
    let byte_offset = slot.offset_bytes.checked_add(storage.base_offset_bytes)
        .ok_or(CpuNativeError::InvalidShape)?;
    if byte_offset % core::mem::align_of::<f32>() != 0 {
        return Err(CpuNativeError::UnsupportedLayout);
    }
    let binding = bindings.iter().find(|item| item.storage() == storage_id)
        .ok_or(CpuNativeError::MissingBinding)?;
    // Validate the real external binding's element type even when the unsafe
    // backend entry is invoked directly instead of through PreparedExecution.
    if binding.dtype() != DType::F32 {
        return Err(CpuNativeError::UnsupportedLayout);
    }
    let handle = match binding.buffer() {
        BufferHandle::Host(handle) => handle,
        _ => return Err(CpuNativeError::UnsupportedMemory),
    };
    if handle.bytes > isize::MAX as usize
        || (handle.bytes != 0 && (handle.address == 0
            || handle.address.checked_add(handle.bytes).is_none()))
    {
        return Err(CpuNativeError::InvalidShape);
    }
    if byte_offset.checked_add(bytes).is_none_or(|total| total > handle.bytes) {
        return Err(CpuNativeError::InvalidShape);
    }
    // The physical storage may include unused padding beyond the logical
    // payload. Binding must prove that the *entire* frozen region fits both
    // its slot and the real external buffer before a noalias promise is used.
    if !physical_storage_fits(slot.offset_bytes, slot.bytes, storage.physical_bytes, handle.bytes) {
        return Err(CpuNativeError::InvalidShape);
    }
    let physical_start = if storage.physical_bytes == 0 {
        0
    } else {
        handle.address.checked_add(slot.offset_bytes).ok_or(CpuNativeError::InvalidShape)?
    };
    // Exposed-provenance round trip is necessary because StorageBinding's
    // published host-handle ABI uses integer addresses. Never construct a
    // Rust reference from these pointers or reuse them past the bound lease.
    let address = if count == 0 {
        core::ptr::NonNull::<f32>::dangling().as_ptr()
    } else {
        let address = handle.address.checked_add(byte_offset).ok_or(CpuNativeError::InvalidShape)?;
        if address == 0 || address % core::mem::align_of::<f32>() != 0 {
            return Err(CpuNativeError::UnsupportedLayout);
        }
        core::ptr::with_exposed_provenance_mut::<f32>(address)
    };
    Ok(Span {
        ptr: address,
        len: count,
        shape: tensor.shape,
        physical: PhysicalRange { start: physical_start, bytes: storage.physical_bytes },
    })
}

/// The full region occupied by the storage must fit its frozen slot and
/// the actual backing allocation, even when the logical data are smaller.
/// All addition is checked; this helper runs only during BIND.
#[inline]
fn physical_storage_fits(
    slot_offset: usize, slot_bytes: usize, storage_bytes: usize, bound_bytes: usize,
) -> bool {
    storage_bytes <= slot_bytes
        && slot_offset.checked_add(storage_bytes)
            .is_some_and(|end| end <= bound_bytes)
}

/// Validate a C-order logical payload inside a larger physical buffer.
/// This is done at bind, never at replay. In particular, unaligned or
/// overflowing offsets cannot be converted into a raw f32 pointer.
#[inline]
fn contiguous_payload_fits(physical_bytes: usize, base_bytes: usize, data_bytes: usize) -> bool {
    base_bytes % core::mem::size_of::<f32>() == 0
        && base_bytes.checked_add(data_bytes).is_some_and(|end| end <= physical_bytes)
}

/// Only one caller-thread worker covering the FULL flattened output is
/// supported by this native baseline. A partial frozen work range cannot be
/// silently widened to the complete output at runtime.
#[inline]
fn whole_output_work(work: WorkRange, output_len: usize) -> bool {
    work.begin == 0 && u64::try_from(output_len).ok() == Some(work.end)
}

/// This provider is statically compiled by this package's Cargo RELEASE
/// profile; its codegen identity cannot be dynamically changed by a plan.
/// The generic frozen-plan validator permits *other* valid AOT strategies
/// (PGO, native objects, panic=immediate-abort, etc.). Those are not the
/// executable artifact shipped by this restricted built-in backend.
///
/// External image hashes are still caller-supplied and not authenticated here.
/// A genuinely attested release must also verify the precise compiler build,
/// target triple/features, binary bytes and signature at catalog load.
/// This is a cold admission guard, not replay dispatch. Test/bench builds
/// normally unwind even though the release profile requests panic=abort.
/// Do not label a test image as the production native artifact.
#[inline]
fn builtin_release_image_admissible() -> bool {
    cfg!(panic = "abort")
        && !cfg!(debug_assertions)
        && env!("MOMIRU_BUILD_PROFILE") == "release"
        && env!("MOMIRU_BUILD_OPT_LEVEL") == "3"
        && env!("MOMIRU_BUILD_STANDARD_RELEASE") == "true"
        // No external exact rustc toolchain pin means no attested release.
        && env!("MOMIRU_BUILD_PINNED") == "true"
        && env!("MOMIRU_BUILD_OVERRIDDEN") == "false"
}

/// Probe the built-in provider's *build-profile* gate from a real, optimized
/// release executable. `cargo test` (even `--release`) can use a distinct panic
/// strategy, so a test-only assertion cannot verify the positive release case.
///
/// A `true` return means only that this **linked Rust crate** has the expected
/// profile metadata; it does not authenticate any image, attest a native
/// symbol, verify the hardware, or show that a frozen plan can be replayed.
/// The production binder must still perform every independent AOT check.
#[doc(hidden)]
#[inline]
pub fn builtin_cpu_release_profile_matches() -> bool {
    builtin_release_image_admissible()
}

/// Check a few actual linked **Release-mode** scalar/AVX2 entry points with
/// independently computed FP32 fixtures (including non-multiple-of-eight
/// tails). This does not construct a frozen plan and is not a speed test.
/// Nothing here executes on the ordinary prepare/bind/replay path.
#[doc(hidden)]
#[inline(never)]
pub fn builtin_cpu_release_kernel_selfcheck() -> bool {
    if !builtin_release_image_admissible() { return false; }
    // black_box prevents a whole-program optimizer from trivially replacing
    // the fixed-vector checks with a compile-time Boolean constant.
    let a = std::hint::black_box([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = std::hint::black_box([2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let mut out = [f32::NAN; 9];
    let command = CpuCommand {
        a: a.as_ptr(), b: b.as_ptr(), out: out.as_mut_ptr(),
        len: 9, m: 0, k: 0, n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: add_f32,
    };
    unsafe { add_f32(&command); }
    let expected = [3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];
    if out.iter().zip(expected).any(|(actual, reference)| actual.to_bits() != reference.to_bits()) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        out.fill(f32::NAN);
        unsafe { add_avx2(&command); }
        if out.iter().zip(expected).any(|(actual, reference)| actual.to_bits() != reference.to_bits()) {
            return false;
        }
    }

    // Check the selected 256-bit divide path including the ninth-element tail;
    // compare exact f32 bits against the independent scalar instruction path.
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { div_f32(&command); }
        let scalar_div_bits = out.map(f32::to_bits);
        out.fill(f32::NAN);
        unsafe { div_avx2(&command); }
        if out.map(f32::to_bits) != scalar_div_bits { return false; }
    }

    // Sign is a separate frozen entry: signed zeros must not turn into
    // +/-1 as they would with f32::signum, and NaN must not become zero.
    let sign_input = std::hint::black_box([-0.0_f32, 0.0, -3.0, 2.0,
        f32::NEG_INFINITY, f32::INFINITY, f32::NAN]);
    let mut sign_output = [42.0_f32; 7];
    let sign_cmd = CpuCommand {
        a: sign_input.as_ptr(), b: ptr::null(), out: sign_output.as_mut_ptr(),
        len: sign_input.len(), m: 0, k: 0, n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: sign_f32,
    };
    unsafe { sign_f32(&sign_cmd); }
    if sign_output[0].to_bits() != (-0.0_f32).to_bits()
        || sign_output[1].to_bits() != 0.0_f32.to_bits()
        || sign_output[2] != -1.0 || sign_output[3] != 1.0
        || sign_output[4] != -1.0 || sign_output[5] != 1.0
        || !sign_output[6].is_nan()
    { return false; }

    let matrix_a = std::hint::black_box([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let matrix_b = std::hint::black_box([7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let mut matrix_out = [f32::NAN; 4];
    let matrix = CpuCommand {
        a: matrix_a.as_ptr(), b: matrix_b.as_ptr(), out: matrix_out.as_mut_ptr(),
        len: 4, m: 2, k: 3, n: 2,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: gemm_dot_f32,
    };
    let expected_matrix = [58.0f32, 64.0, 139.0, 154.0];
    unsafe { gemm_dot_f32(&matrix); }
    if matrix_out != expected_matrix { return false; }
    unsafe { gemm_axpy_f32(&matrix); }
    if matrix_out != expected_matrix { return false; }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { gemm_avx2(&matrix); }
        if matrix_out != expected_matrix { return false; }
    }
    // These compatibility modes have distinct names and numeric behavior.
    // Verify the linked Release image, not just source-level Rust unit tests.
    let special = std::hint::black_box([-4.0_f32, -1.0, -0.0, 0.0, 0.25, 1.0, 4.0, f32::NAN, f32::INFINITY]);
    let mut special_out = [f32::NAN; 9];
    let special_cmd = CpuCommand {
        a: special.as_ptr(), b: core::ptr::null(), out: special_out.as_mut_ptr(),
        len: special_out.len(), m: 0, k: 0, n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: rsqrt_nonpositive_to_zero_f32,
    };
    unsafe { rsqrt_nonpositive_to_zero_f32(&special_cmd); }
    for i in 0..=3 {
        if special_out[i].to_bits() != 0.0_f32.to_bits() { return false; }
    }
    if special_out[4] != 2.0 || special_out[5] != 1.0 || special_out[6] != 0.5
        || !special_out[7].is_nan() || special_out[8] != 0.0
    {
        return false;
    }
    unsafe { log_nonpositive_to_neg87_f32(&special_cmd); }
    for i in 0..=3 {
        if special_out[i].to_bits() != (-87.0_f32).to_bits() { return false; }
    }
    if special_out[5].to_bits() != 0.0_f32.to_bits() || !special_out[7].is_nan() {
        return false;
    }
    unsafe { sigmoid_f32(&special_cmd); }
    if special_out[3].to_bits() != 0.5_f32.to_bits()
        || special_out[8].to_bits() != 1.0_f32.to_bits()
        || !special_out[7].is_nan()
    { return false; }
    unsafe { relu_f32(&special_cmd); }
    if special_out[0].to_bits() != 0.0_f32.to_bits()
        || special_out[2].to_bits() != 0.0_f32.to_bits()
        || special_out[6].to_bits() != 4.0_f32.to_bits()
        || !special_out[7].is_nan()
    { return false; }
    unsafe { round_f32(&special_cmd); }
    if special_out[2].to_bits() != (-0.0_f32).to_bits()
        || special_out[4].to_bits() != 0.0_f32.to_bits()
        || special_out[6] != 4.0 || !special_out[7].is_nan()
    { return false; }
    unsafe { gelu_tanh_approx_f32(&special_cmd); }
    if special_out[2].to_bits() != (-0.0_f32).to_bits()
        || special_out[3].to_bits() != 0.0_f32.to_bits()
        || special_out[8] != f32::INFINITY || !special_out[7].is_nan()
    { return false; }
    // These two explicit erf-based operations must also work in the actual
    // linked Release image. Compare against independently rounded f32 values,
    // and never mistake the tanh-GELU approximation for erf-GELU.
    unsafe { erf_f32(&special_cmd); }
    if (special_out[1] + 0.842_700_8_f32).abs() > 3.0e-7
        || special_out[2].to_bits() != (-0.0_f32).to_bits()
        || special_out[3].to_bits() != 0.0_f32.to_bits()
        || (special_out[5] - 0.842_700_8_f32).abs() > 3.0e-7
        || special_out[8] != 1.0 || !special_out[7].is_nan()
    { return false; }
    unsafe { gelu_erf_f32(&special_cmd); }
    if (special_out[0] + 0.000_126_684_97_f32).abs() > 1.0e-8
        || special_out[2].to_bits() != (-0.0_f32).to_bits()
        || special_out[3].to_bits() != 0.0_f32.to_bits()
        || (special_out[5] - 0.841_344_8_f32).abs() > 3.0e-7
        || special_out[8] != f32::INFINITY || !special_out[7].is_nan()
    { return false; }
    let zeros = std::hint::black_box([0.0_f32, -0.0]);
    let mut minmax = [f32::NAN];
    let minmax_cmd = CpuCommand {
        a: zeros.as_ptr(), b: ptr::null(), out: minmax.as_mut_ptr(),
        len: 1, m: 0, k: zeros.len(), n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: minimum_all_f32,
    };
    unsafe { minimum_all_f32(&minmax_cmd); }
    if minmax[0].to_bits() != (-0.0_f32).to_bits() { return false; }
    unsafe { maximum_all_f32(&minmax_cmd); }
    if minmax[0].to_bits() != 0.0_f32.to_bits() { return false; }

    // Exercise actual release-mode scalar immediates and a scaled norm.
    // No new startup check is inserted into the normal hot replay path.
    let mut scalar = special_cmd;
    scalar.scalar0 = 1.5;
    unsafe { scalar_power_three_halves_f32(&scalar); }
    if special_out[0] != 0.0 || special_out[6] != 8.0
        || special_out[7] != 0.0 || special_out[8] != f32::INFINITY
    { return false; }
    scalar.scalar0 = -1.0;
    scalar.scalar1 = 2.0;
    unsafe { scalar_clamp_f32(&scalar); }
    if special_out[0] != -1.0 || special_out[6] != 2.0
        || !special_out[7].is_nan() || special_out[8] != 2.0
    { return false; }
    let norm_source = std::hint::black_box([3.0_f32, 4.0]);
    let mut norm_result = [f32::NAN];
    let norm = CpuCommand {
        a: norm_source.as_ptr(), b: ptr::null(), out: norm_result.as_mut_ptr(),
        len: 1, m: 0, k: norm_source.len(), n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: l2_norm_all_f32,
    };
    unsafe { mean_all_f32(&norm); }
    if norm_result[0] != 3.5 { return false; }
    // Overflow in an intermediate FP32 sum must not turn a finite mean
    // into infinity in the linked Release binary.
    let large_mean_input = std::hint::black_box([f32::MAX, f32::MAX]);
    let mut large_mean_cmd = norm;
    large_mean_cmd.a = large_mean_input.as_ptr();
    unsafe { mean_all_f32(&large_mean_cmd); }
    if norm_result[0].to_bits() != f32::MAX.to_bits() { return false; }
    unsafe { l1_norm_all_f32(&norm); }
    if norm_result[0] != 7.0 { return false; }
    unsafe { l2_norm_all_f32(&norm); }
    if norm_result[0] != 5.0 { return false; }
    unsafe { linf_norm_all_f32(&norm); }
    if norm_result[0] != 4.0 { return false; }
    unsafe { variance_population_all_f32(&norm); }
    if norm_result[0] != 0.25 { return false; }
    unsafe { stddev_population_all_f32(&norm); }
    if norm_result[0] != 0.5 { return false; }
    unsafe { logsumexp_all_f32(&norm); }
    if (norm_result[0] - (4.0 + (1.0 + (-1.0f32).exp()).ln())).abs() > 1.0e-6 {
        return false;
    }
    // Verify the *linked* release image does not reinstate short reduction
    // cancellation or the f32 intermediate product overflow bug.
    let short_reduction = std::hint::black_box([100_000_000.0f32, 1.0, -100_000_000.0]);
    let mut short_cmd = norm;
    short_cmd.a = short_reduction.as_ptr();
    short_cmd.k = short_reduction.len();
    unsafe { sum_all_f32(&short_cmd); }
    if norm_result[0].to_bits() != 1.0f32.to_bits() { return false; }
    unsafe { mean_all_f32(&short_cmd); }
    if norm_result[0].to_bits() != ((1.0f64 / 3.0) as f32).to_bits() { return false; }
    let short_l1 = std::hint::black_box([16_777_216.0f32, 1.0, 1.0]);
    short_cmd.a = short_l1.as_ptr();
    unsafe { l1_norm_all_f32(&short_cmd); }
    if norm_result[0].to_bits() != 16_777_218.0f32.to_bits() { return false; }
    let rescaled_product = std::hint::black_box([1.0e20f32, 1.0e20, 1.0e-40]);
    short_cmd.a = rescaled_product.as_ptr();
    unsafe { product_all_f32(&short_cmd); }
    let expected_product = (f64::from(rescaled_product[0])
        * f64::from(rescaled_product[1]) * f64::from(rescaled_product[2])) as f32;
    if norm_result[0].to_bits() != expected_product.to_bits() { return false; }

    // Release image smoke checks for two new rank/dtype-preserving AOT
    // families. Neither check changes the input data nor hot replay path.
    let axis_source = std::hint::black_box([1000.0f32, 1001.0, 1.0, 2.0]);
    let mut axis_output = [f32::NAN; 4];
    let axis_cmd = CpuCommand {
        a: axis_source.as_ptr(), b: ptr::null(), out: axis_output.as_mut_ptr(),
        len: 4, m: 2, k: 2, n: 1,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: softmax_axis_f32,
    };
    unsafe { softmax_axis_f32(&axis_cmd); }
    if (axis_output[0] - 0.26894143).abs() > 1.0e-6
        || (axis_output[1] - 0.7310586).abs() > 1.0e-6
        || (axis_output[2] - 0.26894143).abs() > 1.0e-6
        || (axis_output[3] - 0.7310586).abs() > 1.0e-6
    { return false; }
    unsafe { log_softmax_axis_f32(&axis_cmd); }
    if (axis_output[0] + 1.3132616).abs() > 2.0e-6
        || (axis_output[1] + 0.3132617).abs() > 2.0e-6
    { return false; }
    let scan_source = std::hint::black_box([1.0_f32, 2.0, 3.0, 4.0]);
    let mut scan_output = [f32::NAN; 4];
    let scan_cmd = CpuCommand {
        a: scan_source.as_ptr(), b: ptr::null(), out: scan_output.as_mut_ptr(),
        len: 4, m: 1, k: 4, n: 1,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: scan_axis_f32::<0, true>,
    };
    unsafe { scan_axis_f32::<0, true>(&scan_cmd); }
    if scan_output != [1.0, 3.0, 6.0, 10.0] { return false; }
    unsafe { scan_axis_f32::<0, false>(&scan_cmd); }
    if scan_output != [0.0, 1.0, 3.0, 6.0] { return false; }
    let norm_source = std::hint::black_box([3.0_f32, 4.0]);
    let mut norm_output = [f32::NAN; 2];
    let norm_cmd = CpuCommand {
        a: norm_source.as_ptr(), b: ptr::null(), out: norm_output.as_mut_ptr(),
        len: 2, m: 1, k: 2, n: 1,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: normalize_suffix_f32::<2>,
    };
    unsafe { normalize_suffix_f32::<2>(&norm_cmd); }
    if norm_output[0].to_bits() != 0.6f32.to_bits()
        || norm_output[1].to_bits() != 0.8f32.to_bits()
    { return false; }
    // Release-linked nonconsecutive-axis paths are separate artifacts; check
    // their actual copied FP32 bits and numerical result, not only the BIND
    // branch or the unqualified unit-test harness.
    let perm_source = std::hint::black_box([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let mut perm_output = [f32::NAN; 12];
    let perm_axes = ArbitraryAxes {
        dims: [2, 3, 2, 1, 1, 1, 1, 1], rank: 3, mask: 0b101,
    };
    let perm_cmd = CpuCommand {
        a: perm_source.as_ptr(), b: ptr::null(), out: perm_output.as_mut_ptr(),
        len: 12, m: 3, k: 4, n: 1,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 1e-5, scalar1: 0.0, arbitrary_axes: Some(perm_axes),
        sort_scratch: ptr::null_mut(),
        entry: flip_arbitrary_axes_f32,
    };
    unsafe { flip_arbitrary_axes_f32(&perm_cmd); }
    if perm_output != [8.0, 7.0, 10.0, 9.0, 12.0, 11.0,
        2.0, 1.0, 4.0, 3.0, 6.0, 5.0] { return false; }
    unsafe { normalize_arbitrary_axes_f32::<2>(&perm_cmd); }
    // The independently computed L2 norm over axes [0,2] at j=0 is
    // sqrt(1^2 + 2^2 + 7^2 + 8^2 + epsilon) = sqrt(118+epsilon).
    let denominator = (118.0f64 + f64::from(perm_cmd.scalar0)).sqrt();
    if (f64::from(perm_output[0]) - 1.0 / denominator).abs() > 1.0e-6
        || (f64::from(perm_output[1]) - 2.0 / denominator).abs() > 1.0e-6
        || (f64::from(perm_output[6]) - 7.0 / denominator).abs() > 1.0e-6
        || (f64::from(perm_output[7]) - 8.0 / denominator).abs() > 1.0e-6
    { return false; }
    // Linked release provider: stable value sorting must preserve both
    // distinct NaN payloads and +/-0 input order, with the allocation
    // already prepared by BIND (stack-owned only in this smoke check).
    let sorted_source = std::hint::black_box([
        f32::from_bits(0x7fc0_0011), -0.0, 0.0,
        f32::from_bits(0x7fc0_0022), 1.0, -1.0,
    ]);
    let mut sorted_output = [0.0_f32; 6];
    let mut sort_buffer = [SortRecord { bits: 0, ordinal: 0 }; 6];
    let mut sort_cmd = perm_cmd;
    sort_cmd.a = sorted_source.as_ptr();
    sort_cmd.out = sorted_output.as_mut_ptr();
    sort_cmd.len = 6;
    sort_cmd.m = 1;
    sort_cmd.k = 6;
    sort_cmd.n = 1;
    sort_cmd.sort_scratch = sort_buffer.as_mut_ptr();
    unsafe { sort_values_axis_f32::<false, false>(&sort_cmd); }
    if sorted_output.map(f32::to_bits) != [(-1.0_f32).to_bits(),
        (-0.0_f32).to_bits(), 0.0_f32.to_bits(), 1.0_f32.to_bits(),
        0x7fc0_0011, 0x7fc0_0022] { return false; }
    true
}

#[inline]
fn builtin_codegen_matches_release(codegen: RustCodegenPlan, avx2: bool) -> bool {
    codegen.rustc_version == env!("MOMIRU_BUILD_RUSTC")
        && codegen.target_triple == env!("MOMIRU_BUILD_TARGET")
        // Generic is the package's default codegen target. Custom -C target-cpu
        // and -C target-feature require a separate attested native artifact.
        && codegen.target_cpu == "generic"
        && codegen.target_features.is_empty()
        && codegen.vectorize_loops
        && codegen.vectorize_slp
        && !codegen.cold_error_paths
        && codegen.float_mode == RustFloatMode::Strict
        && codegen.source == (if avx2 {
            CpuKernelSource::StdArchIntrinsics
        } else {
            CpuKernelSource::AutoVectorizedRust
        })
        && codegen.panic_strategy == RustPanicStrategy::Abort
        && codegen.opt_level == 3
        && codegen.lto == LtoMode::Thin
        && codegen.codegen_units == 1
        && codegen.pgo == PgoMode::None
        && codegen.pgo_profile_hash_hi == 0
        && codegen.pgo_profile_hash_lo == 0
        && codegen.post_link_optimization == PostLinkOptimization::None
        && codegen.post_link_profile_hash_hi == 0
        && codegen.post_link_profile_hash_lo == 0
        && !codegen.nightly_codegen
        && !codegen.linker_plugin_lto
        && codegen.tune_cpu.is_empty()
        && codegen.code_model.is_empty()
        && codegen.relocation_model.is_empty()
        && !codegen.incremental
        && !codegen.overflow_checks
        && !codegen.debug_assertions
        && !codegen.force_frame_pointers
        && codegen.llvm_args_hash_hi == 0
        && codegen.llvm_args_hash_lo == 0
}

/// Pairwise disjointness over the complete frozen storage regions, not just
/// the flat arrays accessed by a particular kernel.
#[inline]
fn physical_io_is_disjoint(a: Option<Span>, b: Option<Span>, output: Span) -> bool {
    a.is_none_or(|lhs| lhs.physical.disjoint_from(output.physical))
        && b.is_none_or(|rhs| rhs.physical.disjoint_from(output.physical))
        && a.zip(b).is_none_or(|(lhs, rhs)| lhs.physical.disjoint_from(rhs.physical))
}

/// Resolve any contiguous rank-1..8 axis to the immutable (outer, axis, inner)
/// iteration geometry. A zero-size non-axis dimension is allowed and makes
/// the *entire* replay a no-op. Softmax rejects a zero-size selected axis;
/// scans also admit it as a no-op. No shape changes or dtype conversions.
fn contiguous_axis_geometry(shape: StaticShape, axis: usize, allow_empty_axis: bool)
    -> Result<(usize, usize, usize), CpuNativeError>
{
    let rank = shape.rank();
    if axis >= rank { return Err(CpuNativeError::InvalidShape); }
    let axis_length = shape.dim(axis).ok_or(CpuNativeError::InvalidShape)?;
    if axis_length == 0 && !allow_empty_axis {
        return Err(CpuNativeError::InvalidShape);
    }
    // A genuinely empty tensor has no payload to address. Its other extents
    // may be individually enormous, even if their raw product would overflow.
    // Return inert geometry instead of rejecting a valid zero-element view.
    if shape.element_count() == Some(0) {
        return Ok((0, axis_length, 0));
    }
    let mut outer = 1usize;
    let mut inner = 1usize;
    let mut axis_length = 0usize;
    for dimension in 0..rank {
        let extent = shape.dim(dimension).ok_or(CpuNativeError::InvalidShape)?;
        if dimension < axis {
            outer = outer.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        } else if dimension == axis {
            axis_length = extent;
        } else {
            inner = inner.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        }
    }
    if axis_length == 0 && !allow_empty_axis {
        return Err(CpuNativeError::InvalidShape);
    }
    // Strictly prove every replayed pointer displacement is in the bound span;
    // never infer it from axis indices alone or rely on unchecked wrapping.
    let elements = outer.checked_mul(axis_length)
        .and_then(|value| value.checked_mul(inner))
        .ok_or(CpuNativeError::InvalidShape)?;
    if elements != shape.element_count().ok_or(CpuNativeError::InvalidShape)? {
        return Err(CpuNativeError::InvalidShape);
    }
    Ok((outer, axis_length, inner))
}

/// Materializing a reversal on EVERY axis of a consecutive block is exactly
/// the same as reversing the flattened index inside that block (singleton
/// axes are neutral). This supports single and multi-axis flips using the
/// SAME allocation-free, memcpy-based native entry. Nonconsecutive, nontrivial
/// axes require a distinct mapping kernel; never approximate their effect.
fn flip_block_geometry(shape: StaticShape, axes: AxisList)
    -> Result<(usize, usize, usize), CpuNativeError>
{
    let rank = shape.rank();
    axes.validate_for_rank(rank).map_err(CpuNativeError::InvalidPlan)?;
    if axes.len == 0 { return Err(CpuNativeError::InvalidShape); }
    let elements = shape.element_count().ok_or(CpuNativeError::InvalidShape)?;
    // No bytes are accessed in an empty tensor: huge nonzero extents may be
    // safely ignored, but the complete axis selection must still be legal.
    if elements == 0 { return Ok((0, 0, 0)); }
    let mut first = rank;
    let mut last = 0usize;
    let mut flipped = [false; 8];
    for index in 0..usize::from(axes.len) {
        let axis = usize::from(axes.axes[index]);
        if shape.dim(axis).ok_or(CpuNativeError::InvalidShape)? > 1 {
            flipped[axis] = true;
            first = first.min(axis);
            last = last.max(axis);
        }
    }
    if first == rank {
        // All selected dimensions were size-one: a raw copy is the flip.
        return Ok((1, 1, elements));
    }
    let mut outer = 1usize;
    let mut block = 1usize;
    let mut inner = 1usize;
    for axis in 0..rank {
        let extent = shape.dim(axis).ok_or(CpuNativeError::InvalidShape)?;
        if axis >= first && axis <= last {
            if extent > 1 && !flipped[axis] {
                return Err(CpuNativeError::UnsupportedOperation);
            }
            block = block.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        } else if axis < first {
            outer = outer.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        } else {
            inner = inner.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        }
    }
    if outer.checked_mul(block).and_then(|value| value.checked_mul(inner)) != Some(elements) {
        return Err(CpuNativeError::InvalidShape);
    }
    Ok((outer, block, inner))
}

/// Record the exact contiguous row-major logical shape and axis mask once.
/// `element_count` is already checked for every native IO span at BIND.
fn arbitrary_axes_geometry(shape: StaticShape, mask: u8)
    -> Result<ArbitraryAxes, CpuNativeError>
{
    let rank = shape.rank();
    AxisMask(mask).validate_for_rank(rank).map_err(CpuNativeError::InvalidPlan)?;
    if rank == 0 || mask == 0 { return Err(CpuNativeError::InvalidShape); }
    shape.element_count().ok_or(CpuNativeError::InvalidShape)?;
    let mut dims = [1usize; 8];
    for (axis, dim) in dims.iter_mut().enumerate().take(rank) {
        *dim = shape.dim(axis).ok_or(CpuNativeError::InvalidShape)?;
    }
    Ok(ArbitraryAxes { dims, rank: rank as u8, mask })
}

/// Generic normalized axes include every explicitly selected dimension;
/// none may have zero extent, even if some other dimension is empty.
fn normalized_arbitrary_geometry(shape: StaticShape, axes: AxisMask)
    -> Result<(usize, usize, ArbitraryAxes), CpuNativeError>
{
    let g = arbitrary_axes_geometry(shape, axes.0)?;
    for axis in 0..usize::from(g.rank) {
        if (g.mask & (1_u8 << axis)) != 0 && g.dims[axis] == 0 {
            return Err(CpuNativeError::InvalidShape);
        }
    }
    let numel = shape.element_count().ok_or(CpuNativeError::InvalidShape)?;
    if numel == 0 { return Ok((0, 0, g)); }
    let (mut groups, mut count) = (1usize, 1usize);
    for axis in 0..usize::from(g.rank) {
        if (g.mask & (1_u8 << axis)) != 0 {
            count = count.checked_mul(g.dims[axis]).ok_or(CpuNativeError::InvalidShape)?;
        } else {
            groups = groups.checked_mul(g.dims[axis]).ok_or(CpuNativeError::InvalidShape)?;
        }
    }
    if groups.checked_mul(count) != Some(numel) { return Err(CpuNativeError::InvalidShape); }
    Ok((groups, count, g))
}

/// A compact rank-preserving normalization kernel handles only a trailing
/// contiguous set of axes. Other axis subsets must bind a distinct, verified
/// provider instead of running with incorrect indexing or shape coercion.
fn normalized_suffix_geometry(shape: StaticShape, axes: AxisMask)
    -> Result<(usize, usize), CpuNativeError>
{
    let rank = shape.rank();
    axes.validate_for_rank(rank).map_err(CpuNativeError::InvalidPlan)?;
    if rank == 0 || axes.is_empty() { return Err(CpuNativeError::InvalidShape); }
    let first = axes.0.trailing_zeros() as usize;
    let valid_mask = if rank == 8 { u8::MAX } else { ((1u16 << rank) - 1) as u8 };
    if axes.0 != (valid_mask & (u8::MAX << first)) {
        return Err(CpuNativeError::UnsupportedOperation);
    }
    for axis in first..rank {
        if shape.dim(axis) == Some(0) { return Err(CpuNativeError::InvalidShape); }
    }
    if shape.element_count() == Some(0) {
        // Empty tensors can have huge other dimensions: do not overflow
        // an inactive loop's geometry when it cannot access any elements.
        return Ok((0, 0));
    }
    let mut groups = 1usize;
    let mut normalized_count = 1usize;
    for axis in 0..rank {
        let extent = shape.dim(axis).ok_or(CpuNativeError::InvalidShape)?;
        if axis < first {
            groups = groups.checked_mul(extent).ok_or(CpuNativeError::InvalidShape)?;
        } else {
            normalized_count = normalized_count.checked_mul(extent)
                .ok_or(CpuNativeError::InvalidShape)?;
        }
    }
    if groups.checked_mul(normalized_count) != shape.element_count() {
        return Err(CpuNativeError::InvalidShape);
    }
    Ok((groups, normalized_count))
}

/// GroupNorm normalizes each (sample, group-of-channels) over the group's
/// channels and all spatial axes. The IR requires the batch axis to be zero,
/// and validates divisibility of the logical channel dimension by groups.
/// A singleton group or a channel axis immediately after batch has a truly
/// contiguous physical group, so its existing suffix kernel is faster.
fn group_normalization_geometry(
    shape: StaticShape, channel_axis: u8, groups: u32,
) -> Result<(usize, usize, ArbitraryAxes, bool), CpuNativeError> {
    let rank = shape.rank();
    let ch = usize::from(channel_axis);
    if rank < 2 || ch == 0 || ch >= rank || groups == 0 {
        return Err(CpuNativeError::InvalidShape);
    }
    let descriptor = arbitrary_axes_geometry(shape, 1_u8 << ch)?;
    let count_groups = usize::try_from(groups).map_err(|_| CpuNativeError::InvalidShape)?;
    let channels = descriptor.dims[ch];
    if channels == 0 || channels % count_groups != 0 {
        return Err(CpuNativeError::InvalidShape);
    }
    // Every normalized axis is nonempty. Only the batch axis may be empty.
    for axis in 1..rank {
        if descriptor.dims[axis] == 0 { return Err(CpuNativeError::InvalidShape); }
    }
    let numel = shape.element_count().ok_or(CpuNativeError::InvalidShape)?;
    let use_direct = ch == 1 || count_groups == 1;
    if numel == 0 { return Ok((0, 0, descriptor, use_direct)); }
    let mut group_elements = channels / count_groups;
    for axis in 1..rank {
        if axis != ch {
            group_elements = group_elements.checked_mul(descriptor.dims[axis])
                .ok_or(CpuNativeError::InvalidShape)?;
        }
    }
    let group_count = descriptor.dims[0].checked_mul(count_groups)
        .ok_or(CpuNativeError::InvalidShape)?;
    if group_count.checked_mul(group_elements) != Some(numel) {
        return Err(CpuNativeError::InvalidShape);
    }
    Ok((group_count, group_elements, descriptor, use_direct))
}

fn prepare_node(
    plan: &FrozenExecutionPlan, node: &FrozenNode, kernel: &CpuKernelPlan,
    bindings: &[StorageBinding<'_>], installed: CpuInstallation,
) -> Result<CpuCommand, CpuNativeError> {
    if !matches!(kernel.math_mode, KernelMathMode::StrictFp32 | KernelMathMode::Exact) {
        return Err(CpuNativeError::UnsupportedMathMode);
    }
    // The global frozen-plan validator requires precisely one logical worker
    // for SingleThread, including CallerThread. Requiring ZERO workers here
    // would make every structurally valid CPU plan impossible to bind.
    if kernel.dispatch_mode != CpuDispatchMode::CallerThread
        || kernel.parallel_kind != CpuParallelKind::SingleThread
        || kernel.workers.len() != 1 || !kernel.workspace_slots.is_empty()
        || kernel.library_algorithm.is_some()
        || kernel.pack_a || kernel.pack_b || kernel.persistent_packed_weights
        || kernel.store_policy != StorePolicy::Cached
        || kernel.prefetch_kind != PrefetchKind::None
        || kernel.prefetch_distance_bytes != 0
        || !kernel.memory_access.is_empty()
        || kernel.numa_policy != NumaPolicy::Local
        || kernel.fixed_numa_node != 0
        || kernel.worker_wait != WorkerWaitPolicy::Park
        || kernel.assumptions.assert_unchecked_invariants
        || kernel.pack_alignment_bytes != 0
        // A built-in direct kernel has no tile/pack/unroll/pipeline machinery.
        // Reject any AOT plan promising one rather than silently ignoring it.
        || kernel.vector_unroll != 1 || kernel.scalar_unroll != 1
        || kernel.software_pipeline_stages != 1
        || kernel.tile_m != 1 || kernel.tile_n != 1 || kernel.tile_k != 1
        || kernel.micro_m != 1 || kernel.micro_n != 1 || kernel.micro_k != 1
        || kernel.artifact.kind != ArtifactKind::RustAotFunction
        || kernel.artifact.artifact != BUILTIN_CPU_ARTIFACT
        || kernel.artifact.artifact_hash_hi != installed.artifact_hash_hi
        || kernel.artifact.artifact_hash_lo != installed.artifact_hash_lo
        || kernel.artifact.target_profile_hash_hi != installed.capabilities.hardware_profile_hash_hi
        || kernel.artifact.target_profile_hash_lo != installed.capabilities.hardware_profile_hash_lo
    {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    let features = kernel.required_features;
    let portable = features == FeatureMask(0);
    let avx2 = features == cpu_feature::AVX2;
    if !portable && !avx2 { return Err(CpuNativeError::UnsupportedCpuFeature); }
    // The only vectorized built-in paths are AVX2 add/sub/mul/div and GEMM.
    // A scalar/copy plan must not claim to execute an AVX2-specific artifact.
    let required_simd_bytes: u16 = if avx2 { 32 } else { 0 };
    if kernel.simd_bytes != required_simd_bytes {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    if avx2 {
        if installed.capabilities.hardware.cpu
            .is_none_or(|cpu| !cpu.features.contains(cpu_feature::AVX2)) {
            return Err(CpuNativeError::UnsupportedCpuFeature);
        }
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2") {
            return Err(CpuNativeError::UnsupportedCpuFeature);
        }
        #[cfg(not(target_arch = "x86_64"))]
        return Err(CpuNativeError::UnsupportedCpuFeature);
    }
    if !builtin_release_image_admissible() {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    // validate_cpu_mapping requires Rust codegen metadata for this exact
    // artifact kind; a direct unsafe Backend call must not silently skip it.
    let codegen = kernel.rust_codegen.ok_or(CpuNativeError::UnsupportedArtifact)?;
    if !builtin_codegen_matches_release(codegen, avx2) {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    // "Exact" describes a bitwise output contract. Arithmetic can produce
    // NaN payload/reassociation variation across targets; only a raw copy,
    // axis flip or stable value sort transports each original f32 payload
    // unchanged and deterministically. Indices sort is rejected below, as its
    // output dtype differs from this provider's f32-only ABI.
    if kernel.math_mode == KernelMathMode::Exact
        && !matches!(node.operation,
            Operation::Copy | Operation::Flip { .. } | Operation::Sort(_))
    {
        return Err(CpuNativeError::UnsupportedMathMode);
    }
    let inputs = kernel.io.input_storages;
    let outputs = kernel.io.output_storages;
    if outputs.len() != 1 || inputs.len() > 2 { return Err(CpuNativeError::UnsupportedOperation); }
    let output = resolve_span(plan, outputs[0], bindings)?;
    if bindings.iter().find(|b| b.storage() == outputs[0])
        .is_none_or(|binding| binding.access() != BindingAccess::ReadWrite)
        || plan.storages[outputs[0] as usize].access == StorageAccess::ReadOnly
    {
        return Err(CpuNativeError::MissingBinding);
    }
    for &input_id in inputs {
        if plan.storages.get(input_id as usize)
            .is_none_or(|storage| storage.id != input_id || storage.access == StorageAccess::WriteOnly) {
            return Err(CpuNativeError::MissingBinding);
        }
    }
    let a = if let Some(&id) = inputs.first() { Some(resolve_span(plan, id, bindings)?) } else { None };
    let b = if let Some(&id) = inputs.get(1) { Some(resolve_span(plan, id, bindings)?) } else { None };
    if a.is_some_and(|src| !src.disjoint_from(output))
        || b.is_some_and(|src| !src.disjoint_from(output)) {
        return Err(CpuNativeError::UnsafeAliasing);
    }
    // The AOT noalias promise covers every PHYSICAL IO storage range,
    // including padding. Checking only touched logical payloads can admit
    // overlapping read-only storages whose wider physical regions alias.
    // This extra check is one-time BIND work, never a replay-time branch.
    if kernel.assumptions.noalias_io && !physical_io_is_disjoint(a, b, output) {
        return Err(CpuNativeError::UnsafeAliasing);
    }
    let alignment = usize::from(kernel.assumptions.aligned_access_bytes);
    if alignment > 0 && !alignment.is_power_of_two() {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    if [Some(output), a, b].iter().flatten().any(|span| {
        span.len != 0 && alignment != 0 && (span.ptr as usize) % alignment != 0
    }) {
        return Err(CpuNativeError::UnsupportedLayout);
    }
    let a = a.ok_or(CpuNativeError::UnsupportedOperation)?;
    let mut command = CpuCommand {
        a: a.ptr.cast_const(),
        b: b.map_or(ptr::null(), |span| span.ptr.cast_const()),
        out: output.ptr,
        len: output.len,
        m: 0, k: 0, n: 0,
        transpose_lhs: false, transpose_rhs: false,
        lhs_vector: false, rhs_vector: false,
        scalar0: 0.0, scalar1: 0.0, arbitrary_axes: None, sort_scratch: ptr::null_mut(), entry: copy_f32,
    };
    let (name, entry): (&str, unsafe fn(&CpuCommand)) = match node.operation {
        Operation::Copy if inputs.len() == 1
            && a.shape == output.shape
            && kernel.family == KernelFamily::ContiguousCopy => (BUILTIN_COPY, copy_f32),
        Operation::Elementwise { kind, arity: 2, broadcast_signature: 0 }
            if inputs.len() == 2 && b.is_some_and(|second| second.shape == a.shape)
                && a.shape == output.shape
                && matches!(kernel.family, KernelFamily::ContiguousElementwise | KernelFamily::Elementwise) => {
            match kind {
                ElementwiseKind::Add => if avx2 {
                    (BUILTIN_ADD_AVX2, add_entry_avx2())
                } else { (BUILTIN_ADD, add_f32) },
                ElementwiseKind::Subtract => if avx2 {
                    (BUILTIN_SUB_AVX2, sub_entry_avx2())
                } else { (BUILTIN_SUB, sub_f32) },
                ElementwiseKind::Multiply => if avx2 {
                    (BUILTIN_MUL_AVX2, mul_entry_avx2())
                } else { (BUILTIN_MUL, mul_f32) },
                ElementwiseKind::Divide => if avx2 {
                    (BUILTIN_DIV_AVX2, div_entry_avx2())
                } else { (BUILTIN_DIV, div_f32) },
                ElementwiseKind::DivideZeroToZero if !avx2 =>
                    (BUILTIN_DIV_ZERO_TO_ZERO, div_zero_to_zero_f32),
                ElementwiseKind::Minimum if !avx2 => (BUILTIN_MIN, min_f32),
                ElementwiseKind::Maximum if !avx2 => (BUILTIN_MAX, max_f32),
                ElementwiseKind::Hypot if !avx2 => (BUILTIN_HYPOT, hypot_f32),
                ElementwiseKind::PRelu if !avx2 => (BUILTIN_PRELU, prelu_f32),
                _ => return Err(CpuNativeError::UnsupportedOperation),
            }
        }
        Operation::Elementwise { kind, arity: 1, broadcast_signature: 0 }
            if inputs.len() == 1 && a.shape == output.shape && !avx2
                && matches!(kernel.family, KernelFamily::ContiguousElementwise | KernelFamily::Elementwise) => {
            match kind {
                ElementwiseKind::Negate => (BUILTIN_NEG, neg_f32),
                ElementwiseKind::Absolute => (BUILTIN_ABS, abs_f32),
                ElementwiseKind::Square => (BUILTIN_SQUARE, square_f32),
                ElementwiseKind::Sqrt => (BUILTIN_SQRT, sqrt_f32),
                ElementwiseKind::SqrtNonPositiveToZero =>
                    (BUILTIN_SQRT_NONPOSITIVE_TO_ZERO, sqrt_nonpositive_to_zero_f32),
                ElementwiseKind::Reciprocal =>
                    (BUILTIN_RECIPROCAL, reciprocal_f32),
                ElementwiseKind::ReciprocalZeroToZero =>
                    (BUILTIN_RECIPROCAL_ZERO_TO_ZERO, reciprocal_zero_to_zero_f32),
                ElementwiseKind::Rsqrt => (BUILTIN_RSQRT, rsqrt_f32),
                ElementwiseKind::RsqrtNonPositiveToZero =>
                    (BUILTIN_RSQRT_NONPOSITIVE_TO_ZERO, rsqrt_nonpositive_to_zero_f32),
                ElementwiseKind::Exp => (BUILTIN_EXP, exp_f32),
                ElementwiseKind::Expm1 => (BUILTIN_EXPM1, expm1_f32),
                ElementwiseKind::Log => (BUILTIN_LOG, log_f32),
                ElementwiseKind::LogNonPositiveToNeg87 =>
                    (BUILTIN_LOG_NONPOSITIVE_TO_NEG87, log_nonpositive_to_neg87_f32),
                ElementwiseKind::Log1p => (BUILTIN_LOG1P, log1p_f32),
                ElementwiseKind::Log1pBelowMinusOneToNeg87 =>
                    (BUILTIN_LOG1P_BELOW_MINUS_ONE_TO_NEG87, log1p_below_minus_one_to_neg87_f32),
                ElementwiseKind::Log1pAbs => (BUILTIN_LOG1P_ABS, log1p_abs_f32),
                ElementwiseKind::Sin => (BUILTIN_SIN, sin_f32),
                ElementwiseKind::Cos => (BUILTIN_COS, cos_f32),
                ElementwiseKind::Tan => (BUILTIN_TAN, tan_f32),
                ElementwiseKind::Erf => (BUILTIN_ERF, erf_f32),
                ElementwiseKind::GeluErfExact => (BUILTIN_GELU_ERF, gelu_erf_f32),
                ElementwiseKind::Tanh => (BUILTIN_TANH, tanh_f32),
                ElementwiseKind::Sigmoid => (BUILTIN_SIGMOID, sigmoid_f32),
                ElementwiseKind::Relu => (BUILTIN_RELU, relu_f32),
                ElementwiseKind::Relu6 => (BUILTIN_RELU6, relu6_f32),
                ElementwiseKind::Silu => (BUILTIN_SILU, silu_f32),
                ElementwiseKind::Softplus => (BUILTIN_SOFTPLUS, softplus_f32),
                ElementwiseKind::Softsign => (BUILTIN_SOFTSIGN, softsign_f32),
                ElementwiseKind::HardSwish => (BUILTIN_HARDSWISH, hard_swish_f32),
                ElementwiseKind::Mish => (BUILTIN_MISH, mish_f32),
                ElementwiseKind::Elu => (BUILTIN_ELU, elu_f32),
                ElementwiseKind::Selu => (BUILTIN_SELU, selu_f32),
                ElementwiseKind::Floor => (BUILTIN_FLOOR, floor_f32),
                ElementwiseKind::Ceil => (BUILTIN_CEIL, ceil_f32),
                ElementwiseKind::Round => (BUILTIN_ROUND, round_f32),
                ElementwiseKind::Sign => (BUILTIN_SIGN, sign_f32),
                ElementwiseKind::GeluTanhApprox => (BUILTIN_GELU_TANH, gelu_tanh_approx_f32),
                _ => return Err(CpuNativeError::UnsupportedOperation),
            }
        }
        Operation::ScalarElementwise(spec)
            if inputs.len() == 1 && a.shape == output.shape && !avx2
                && matches!(kernel.family,
                    KernelFamily::ScalarElementwise | KernelFamily::ScalarPowerThreeHalves) => {
            spec.validate().map_err(CpuNativeError::InvalidPlan)?;
            command.scalar0 = f32::from_bits(spec.scalar0_f32_bits);
            command.scalar1 = f32::from_bits(spec.scalar1_f32_bits);
            choose_scalar_entry(spec.kind, spec.scalar0_f32_bits, kernel.family)?
        }
        Operation::Reduction { kind, axes: AxisSelector::All, keep_dims }
            if inputs.len() == 1 && !avx2
                && kernel.family == KernelFamily::RowwiseReduction => {
            if output.len != 1 || (!keep_dims && output.shape != StaticShape::Rank0)
                || (keep_dims && output.shape.rank() != a.shape.rank()) {
                return Err(CpuNativeError::InvalidShape);
            }
            if keep_dims {
                for axis in 0..a.shape.rank() {
                    if output.shape.dim(axis) != Some(1) { return Err(CpuNativeError::InvalidShape); }
                }
            }
            command.k = a.len;
            choose_all_reduction_entry(kind, a.len)?
        }
        // Flip is materialized into a disjoint f32 buffer. The optimized
        // provider accepts a single axis or a consecutive nontrivial block;
        // other selections require a separate mapping implementation.
        Operation::Flip { axes } if inputs.len() == 1 && !avx2
            && a.shape == output.shape
            && matches!(kernel.family, KernelFamily::FlipGeneric | KernelFamily::FlipContiguous2d) => {
            Operation::Flip { axes }.validate_basic().map_err(CpuNativeError::InvalidPlan)?;
            if kernel.family == KernelFamily::FlipContiguous2d
                && a.shape.rank() != 2 {
                return Err(CpuNativeError::UnsupportedOperation);
            }
            match flip_block_geometry(a.shape, axes) {
                Ok((outer, axis_length, inner)) => {
                    command.m = outer;
                    command.k = axis_length;
                    command.n = inner;
                    (BUILTIN_FLIP_AXIS_BLOCK, flip_axis_f32)
                }
                Err(CpuNativeError::UnsupportedOperation)
                    if kernel.family == KernelFamily::FlipGeneric => {
                    let mut mask = 0_u8;
                    for index in 0..usize::from(axes.len) {
                        mask |= 1_u8 << axes.axes[index];
                    }
                    command.arbitrary_axes = Some(arbitrary_axes_geometry(a.shape, mask)?);
                    (BUILTIN_FLIP_GENERAL, flip_arbitrary_axes_f32)
                }
                Err(error) => return Err(error),
            }
        }
        Operation::Softmax { axis, log } if inputs.len() == 1 && !avx2
            && a.shape == output.shape
            && kernel.family == (if log { KernelFamily::FusedLogSoftmax }
                                 else { KernelFamily::FusedSoftmax }) => {
            let (outer, axis_length, inner) =
                contiguous_axis_geometry(a.shape, usize::from(axis), false)?;
            command.m = outer;
            command.k = axis_length;
            command.n = inner;
            if log { (BUILTIN_LOG_SOFTMAX, log_softmax_axis_f32) }
            else { (BUILTIN_SOFTMAX, softmax_axis_f32) }
        }
        Operation::Scan(spec) if inputs.len() == 1 && !avx2
            && spec.output == ScanOutput::Values
            && a.shape == output.shape
            && matches!(kernel.family,
                KernelFamily::ParallelPrefixScan | KernelFamily::SegmentedScan) => {
            Operation::Scan(spec).validate_basic().map_err(CpuNativeError::InvalidPlan)?;
            let (outer, axis_length, inner) =
                contiguous_axis_geometry(a.shape, usize::from(spec.axis), true)?;
            command.m = outer;
            command.k = axis_length;
            command.n = inner;
            match (spec.kind, spec.inclusive) {
                (ScanKind::Sum, true) =>
                    (BUILTIN_SCAN_SUM_INCLUSIVE, scan_axis_f32::<0, true>),
                (ScanKind::Sum, false) =>
                    (BUILTIN_SCAN_SUM_EXCLUSIVE, scan_axis_f32::<0, false>),
                (ScanKind::Product, true) =>
                    (BUILTIN_SCAN_PRODUCT, scan_axis_f32::<1, true>),
                (ScanKind::Minimum, true) =>
                    (BUILTIN_SCAN_MINIMUM, scan_axis_f32::<2, true>),
                (ScanKind::Maximum, true) =>
                    (BUILTIN_SCAN_MAXIMUM, scan_axis_f32::<3, true>),
                (ScanKind::LogSumExp, true) =>
                    (BUILTIN_SCAN_LOGSUMEXP, scan_axis_f32::<4, true>),
                _ => return Err(CpuNativeError::UnsupportedOperation),
            }
        }
        Operation::Sort(spec) if inputs.len() == 1 && !avx2
            && spec.output == SortOutput::Values && spec.stable
            && a.shape == output.shape
            && kernel.family == KernelFamily::ComparisonSort => {
            Operation::Sort(spec).validate_basic()
                .map_err(CpuNativeError::InvalidPlan)?;
            let (outer, axis_length, inner) =
                contiguous_axis_geometry(a.shape, usize::from(spec.axis), true)?;
            command.m = outer;
            command.k = axis_length;
            command.n = inner;
            match (spec.descending, spec.nan_policy) {
                (false, SortNanPolicy::NaNsFirst) =>
                    (BUILTIN_SORT_ASC_NANS_FIRST, sort_values_axis_f32::<false, true>),
                (false, SortNanPolicy::NaNsLast) =>
                    (BUILTIN_SORT_ASC_NANS_LAST, sort_values_axis_f32::<false, false>),
                (true, SortNanPolicy::NaNsFirst) =>
                    (BUILTIN_SORT_DESC_NANS_FIRST, sort_values_axis_f32::<true, true>),
                (true, SortNanPolicy::NaNsLast) =>
                    (BUILTIN_SORT_DESC_NANS_LAST, sort_values_axis_f32::<true, false>),
            }
        }
        Operation::Normalization(spec) if inputs.len() == 1 && !avx2
            && a.shape == output.shape
            && kernel.family == KernelFamily::RowwiseNormalization
            && spec.kind == NormalizationKind::GroupNorm => {
            Operation::Normalization(spec).validate_basic()
                .map_err(CpuNativeError::InvalidPlan)?;
            let (groups, group_size, axes, direct) =
                group_normalization_geometry(a.shape, spec.channel_axis, spec.groups)?;
            command.m = groups;
            command.k = group_size;
            // Only for this group-wise kernel, n is the integer group count.
            command.n = usize::try_from(spec.groups)
                .map_err(|_| CpuNativeError::InvalidShape)?;
            command.scalar0 = f32::from_bits(spec.epsilon_f32_bits);
            command.arbitrary_axes = Some(axes);
            if direct { (BUILTIN_GROUP_NORM_CONTIGUOUS, normalize_suffix_f32::<0>) }
            else { (BUILTIN_GROUP_NORM_CHANNEL_AXIS, normalize_group_channels_f32) }
        }
        Operation::Normalization(spec) if inputs.len() == 1 && !avx2
            && a.shape == output.shape
            && kernel.family == KernelFamily::RowwiseNormalization => {
            Operation::Normalization(spec).validate_basic()
                .map_err(CpuNativeError::InvalidPlan)?;
            // Keep the fast contiguous suffix entry for the common case.
            // Arbitrary but valid axis masks route to an independent, verified
            // allocation-free implementation, selected only at BIND.
            let suffix = normalized_suffix_geometry(a.shape, spec.axes);
            let generic = match suffix {
                Ok((groups, count)) => {
                    command.m = groups;
                    command.k = count;
                    false
                }
                Err(CpuNativeError::UnsupportedOperation) => {
                    let (groups, count, axes) =
                        normalized_arbitrary_geometry(a.shape, spec.axes)?;
                    command.m = groups;
                    command.k = count;
                    command.arbitrary_axes = Some(axes);
                    true
                }
                Err(error) => return Err(error),
            };
            command.n = 1;
            command.scalar0 = f32::from_bits(spec.epsilon_f32_bits);
            match (spec.kind, generic) {
                (NormalizationKind::LayerNorm, false) =>
                    (BUILTIN_LAYERNORM, normalize_suffix_f32::<0>),
                (NormalizationKind::RmsNorm, false) =>
                    (BUILTIN_RMSNORM, normalize_suffix_f32::<1>),
                (NormalizationKind::L2Norm, false) =>
                    (BUILTIN_L2_NORMALIZATION, normalize_suffix_f32::<2>),
                (NormalizationKind::LayerNorm, true) =>
                    (BUILTIN_LAYERNORM_AXES, normalize_arbitrary_axes_f32::<0>),
                (NormalizationKind::RmsNorm, true) =>
                    (BUILTIN_RMSNORM_AXES, normalize_arbitrary_axes_f32::<1>),
                (NormalizationKind::L2Norm, true) =>
                    (BUILTIN_L2_NORMALIZATION_AXES, normalize_arbitrary_axes_f32::<2>),
                // The IR defines BatchNorm as population statistics over
                // every non-channel axis, including the batch axis. With a
                // valid channel_axis != 0, its axis mask is non-suffix.
                (NormalizationKind::BatchNorm, true) =>
                    (BUILTIN_BATCH_NORM_AXES, normalize_arbitrary_axes_f32::<0>),
                // InstanceNorm normalizes each (sample, channel) over its
                // spatial axes. A suffix may use the existing fast kernel.
                (NormalizationKind::InstanceNorm, false) =>
                    (BUILTIN_INSTANCE_NORM_SUFFIX, normalize_suffix_f32::<0>),
                (NormalizationKind::InstanceNorm, true) =>
                    (BUILTIN_INSTANCE_NORM_AXES, normalize_arbitrary_axes_f32::<0>),
                // GroupNorm takes the dedicated branch above, which freezes
                // the channel partition and its exact native entry.
                _ => return Err(CpuNativeError::UnsupportedOperation),
            }
        }
        Operation::MatMul { transpose_lhs, transpose_rhs } if inputs.len() == 2
            && matches!(kernel.family, KernelFamily::CpuSgemmDirect | KernelFamily::CpuSgemmDirectAxpy) => {
            let second = b.ok_or(CpuNativeError::UnsupportedOperation)?;
            let (m,k,n) = matmul_dimensions(a.shape, second.shape, output.shape, transpose_lhs, transpose_rhs)?;
            command.m = m;
            command.k = k;
            command.n = n;
            command.transpose_lhs = transpose_lhs;
            command.transpose_rhs = transpose_rhs;
            command.lhs_vector = a.shape.rank() == 1;
            command.rhs_vector = second.shape.rank() == 1;
            let vectorizable = !transpose_lhs && !transpose_rhs
                && !command.lhs_vector && !command.rhs_vector;
            choose_gemm_entry(kernel.family, avx2, vectorizable)?
        }
        _ => return Err(CpuNativeError::UnsupportedOperation),
    };
    if kernel.artifact.entry != name { return Err(CpuNativeError::UnsupportedArtifact); }
    // Direct built-ins always execute the FULL flattened output domain on the
    // calling thread. No partial sharding, affinity binding or hidden queues.
    // Limit admission to this explicit 0..N work contract until a real worker
    // scheduler and per-family range interpretation exist.
    let work = kernel.workers[0].work;
    if !whole_output_work(work, command.len) {
        return Err(CpuNativeError::UnsupportedArtifact);
    }
    // Do not attach the AVX2 ISA tag to scalar copy or unsupported operations.
    if avx2 && !matches!(node.operation,
        Operation::Elementwise {
            kind: ElementwiseKind::Add | ElementwiseKind::Subtract
                | ElementwiseKind::Multiply | ElementwiseKind::Divide,
            arity: 2, ..
        } | Operation::MatMul { .. })
    {
        return Err(CpuNativeError::UnsupportedCpuFeature);
    }
    command.entry = entry;
    Ok(command)
}

/// Bind-time scalar specialization. The immutable command carries the exact
/// input bit patterns, but the fast replay path does not branch on the scalar
/// operation or exponent. Numeric compatibility is part of each entry name.
fn choose_scalar_entry(
    kind: ScalarElementwiseKind, bits: u32, family: KernelFamily,
) -> Result<(&'static str, unsafe fn(&CpuCommand)), CpuNativeError> {
    if family == KernelFamily::ScalarPowerThreeHalves {
        return if kind == ScalarElementwiseKind::Power && bits == 1.5f32.to_bits() {
            Ok((BUILTIN_SCALAR_POWER_3_2, scalar_power_three_halves_f32))
        } else {
            Err(CpuNativeError::UnsupportedOperation)
        };
    }
    if family != KernelFamily::ScalarElementwise {
        return Err(CpuNativeError::UnsupportedOperation);
    }
    match kind {
        ScalarElementwiseKind::Add => Ok((BUILTIN_SCALAR_ADD, scalar_add_f32)),
        ScalarElementwiseKind::Multiply => Ok((BUILTIN_SCALAR_MUL, scalar_mul_f32)),
        ScalarElementwiseKind::DivideZeroToZero =>
            Ok((BUILTIN_SCALAR_DIV_ZERO, scalar_div_zero_to_zero_f32)),
        ScalarElementwiseKind::LeakyRelu =>
            Ok((BUILTIN_SCALAR_LEAKY_RELU, scalar_leaky_relu_f32)),
        ScalarElementwiseKind::Clamp => Ok((BUILTIN_SCALAR_CLAMP, scalar_clamp_f32)),
        ScalarElementwiseKind::Power => {
            if bits == 1.0f32.to_bits() {
                Ok((BUILTIN_SCALAR_POWER_IDENTITY, scalar_power_identity_f32))
            } else if bits == 2.0f32.to_bits() {
                Ok((BUILTIN_SCALAR_POWER_SQUARE, scalar_power_square_f32))
            } else if bits == 0.5f32.to_bits() {
                Ok((BUILTIN_SCALAR_POWER_SAFE_SQRT, scalar_power_safe_sqrt_f32))
            } else if bits == (-1.0f32).to_bits() {
                Ok((BUILTIN_SCALAR_POWER_SAFE_RECIP, scalar_power_safe_recip_f32))
            } else if bits == 1.5f32.to_bits() {
                Ok((BUILTIN_SCALAR_POWER_3_2, scalar_power_three_halves_f32))
            } else {
                Ok((BUILTIN_SCALAR_POWER_GENERAL, scalar_power_general_f32))
            }
        }
    }
}

/// One cold binding decision. Min/max have NO defined empty-slice identity;
/// sum, product and count-nonzero retain their explicit empty identities.
/// Keeping this decision in a pure helper makes wrong frozen entry selection
/// and unsupported empty reductions testable without a fabricated AOT image.
fn choose_all_reduction_entry(
    kind: ReductionKind, input_len: usize,
) -> Result<(&'static str, unsafe fn(&CpuCommand)), CpuNativeError> {
    match kind {
        ReductionKind::Sum => Ok((BUILTIN_SUM, sum_all_f32)),
        ReductionKind::Product => Ok((BUILTIN_PRODUCT, product_all_f32)),
        ReductionKind::CountNonZero => Ok((BUILTIN_COUNT_NONZERO, count_nonzero_all_f32)),
        ReductionKind::Minimum if input_len != 0 => Ok((BUILTIN_REDUCE_MIN, minimum_all_f32)),
        ReductionKind::Maximum if input_len != 0 => Ok((BUILTIN_REDUCE_MAX, maximum_all_f32)),
        ReductionKind::Mean if input_len != 0 => Ok((BUILTIN_REDUCE_MEAN, mean_all_f32)),
        ReductionKind::L1Norm if input_len != 0 => Ok((BUILTIN_REDUCE_L1, l1_norm_all_f32)),
        ReductionKind::L2Norm if input_len != 0 => Ok((BUILTIN_REDUCE_L2, l2_norm_all_f32)),
        ReductionKind::LInfNorm if input_len != 0 => Ok((BUILTIN_REDUCE_LINF, linf_norm_all_f32)),
        ReductionKind::VariancePopulation if input_len != 0 =>
            Ok((BUILTIN_REDUCE_VARIANCE, variance_population_all_f32)),
        ReductionKind::StdDevPopulation if input_len != 0 =>
            Ok((BUILTIN_REDUCE_STDDEV, stddev_population_all_f32)),
        ReductionKind::LogSumExp if input_len != 0 =>
            Ok((BUILTIN_REDUCE_LOGSUMEXP, logsumexp_all_f32)),
        _ => Err(CpuNativeError::UnsupportedOperation),
    }
}

/// Exact offline GEMM-family admission. Match the label and selected
/// instruction family to the actual linked-in function; a family mismatch
/// is an error, not permission to silently execute another implementation.
fn choose_gemm_entry(
    family: KernelFamily, avx2: bool, vectorizable: bool,
) -> Result<(&'static str, unsafe fn(&CpuCommand)), CpuNativeError> {
    match (family, avx2, vectorizable) {
        (KernelFamily::CpuSgemmDirect, false, _) =>
            Ok((BUILTIN_GEMM, gemm_dot_f32)),
        (KernelFamily::CpuSgemmDirect, true, true) =>
            Ok((BUILTIN_GEMM_AVX2, gemm_entry_avx2())),
        (KernelFamily::CpuSgemmDirectAxpy, false, true) =>
            Ok((BUILTIN_GEMM_AXPY, gemm_axpy_f32)),
        _ => Err(CpuNativeError::UnsupportedOperation),
    }
}

fn matmul_dimensions(
    lhs: StaticShape, rhs: StaticShape, out: StaticShape,
    ta: bool, tb: bool,
) -> Result<(usize,usize,usize), CpuNativeError> {
    let (m,k1) = match lhs {
        StaticShape::Rank1([k]) if !ta => (1,k),
        StaticShape::Rank2([rows, cols]) => if ta { (cols,rows) } else { (rows,cols) },
        _ => return Err(CpuNativeError::UnsupportedOperation),
    };
    let (k2,n) = match rhs {
        StaticShape::Rank1([k]) if !tb => (k,1),
        StaticShape::Rank2([rows, cols]) => if tb { (cols,rows) } else { (rows,cols) },
        _ => return Err(CpuNativeError::UnsupportedOperation),
    };
    if k1 != k2 { return Err(CpuNativeError::InvalidShape); }
    let expected = match (lhs.rank() == 1, rhs.rank() == 1) {
        (true,true) => StaticShape::Rank0,
        (true,false) => StaticShape::Rank1([n]),
        (false,true) => StaticShape::Rank1([m]),
        (false,false) => StaticShape::Rank2([m,n]),
    };
    if out != expected { return Err(CpuNativeError::InvalidShape); }
    Ok((m,k1,n))
}

/// Exact C-order copy; uses memcpy semantics only because the binder proved
/// output and input memory disjoint and exactly equally shaped.
unsafe fn copy_f32(cmd: &CpuCommand) {
    // SAFETY: bounds/non-overlap are proven once by prepare_node.
    unsafe { ptr::copy_nonoverlapping(cmd.a, cmd.out, cmd.len); }
}

macro_rules! simple_binary {
    ($name:ident, $op:tt) => {
        unsafe fn $name(cmd: &CpuCommand) {
            for i in 0..cmd.len {
                // SAFETY: each selected span has precisely cmd.len elements.
                unsafe { *cmd.out.add(i) = *cmd.a.add(i) $op *cmd.b.add(i); }
            }
        }
    };
}
simple_binary!(add_f32, +);
simple_binary!(sub_f32, -);
simple_binary!(mul_f32, *);
simple_binary!(div_f32, /);

/// Compatibility division contract: x / +/-0 -> +0, regardless of x.
/// This intentionally differs from IEEE divide and must have its own AOT ID.
unsafe fn div_zero_to_zero_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe {
            let divisor = *cmd.b.add(i);
            *cmd.out.add(i) = if divisor == 0.0 { 0.0 } else {
                *cmd.a.add(i) / divisor
            };
        }
    }
}

/// IEEE reciprocal and the explicitly separate zero-to-zero compatibility form.
unsafe fn reciprocal_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe { *cmd.out.add(i) = 1.0_f32 / *cmd.a.add(i); }
    }
}
unsafe fn reciprocal_zero_to_zero_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe {
            let value = *cmd.a.add(i);
            *cmd.out.add(i) = if value == 0.0 { 0.0 } else { 1.0 / value };
        }
    }
}
unsafe fn sqrt_nonpositive_to_zero_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe {
            let value = *cmd.a.add(i);
            *cmd.out.add(i) = if value <= 0.0 { 0.0 } else { value.sqrt() };
        }
    }
}

/// NaN propagation, infinity and subnormal semantics follow `f32::hypot`.
/// The kernel always executes the complete C-order tensor without broadcasting.
unsafe fn hypot_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe { *cmd.out.add(i) = (*cmd.a.add(i)).hypot(*cmd.b.add(i)); }
    }
}

/// PReLU's alpha is an entire same-shaped f32 input; broadcasting requires a
/// separate compiled kernel and is never falsely advertised by this entry.
unsafe fn prelu_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe {
            let x = *cmd.a.add(i);
            *cmd.out.add(i) = if x >= 0.0 { x } else { x * *cmd.b.add(i) };
        }
    }
}

// Keep the special-value behavior part of each named AOT entry. In particular,
// the negative/nonpositive compatibility cases are NOT the standard IEEE
// operations and cannot be swapped with them at bind time.
macro_rules! unary_f32 {
    ($name:ident, $x:ident => $expr:expr) => {
        unsafe fn $name(cmd: &CpuCommand) {
            for i in 0..cmd.len {
                let $x = unsafe { *cmd.a.add(i) };
                let y: f32 = $expr;
                unsafe { *cmd.out.add(i) = y; }
            }
        }
    };
}
unary_f32!(rsqrt_f32, x => 1.0_f32 / x.sqrt());
unary_f32!(rsqrt_nonpositive_to_zero_f32, x => {
    if x <= 0.0 { 0.0 } else { 1.0_f32 / x.sqrt() }
});
unary_f32!(exp_f32, x => x.exp());
unary_f32!(expm1_f32, x => x.exp_m1());
unary_f32!(log_f32, x => x.ln());
unary_f32!(log_nonpositive_to_neg87_f32, x => {
    if x <= 0.0 { -87.0_f32 } else { x.ln() }
});
unary_f32!(log1p_f32, x => x.ln_1p());
unary_f32!(log1p_below_minus_one_to_neg87_f32, x => {
    if x <= -1.0 { -87.0_f32 } else { x.ln_1p() }
});
unary_f32!(log1p_abs_f32, x => x.abs().ln_1p());
unary_f32!(sin_f32, x => x.sin());
unary_f32!(cos_f32, x => x.cos());
unary_f32!(tan_f32, x => x.tan());
unary_f32!(tanh_f32, x => x.tanh());

// Evaluation of the mathematical error function does not change the Tensor
// dtype: input and output are f32, and f64 is used only for local scalar
// evaluation of a rational approximation. The coefficients are the classical
// high-accuracy rational erf/erfc approximants (on [0,1], [1,8], [8,27]).
// Keeping erfc as a separate positive-tail primitive avoids subtractive
// cancellation for large negative inputs in erf-based GELU. Unlike the
// ambiguous legacy Gelu opcode this binds a unique, declared formula.
#[inline]
fn horner(x: f64, coeff: &[f64]) -> f64 {
    let mut result = coeff[0];
    for &coefficient in &coeff[1..] {
        result = result * x + coefficient;
    }
    result
}

#[inline]
fn horner_leading_one(x: f64, coeff: &[f64]) -> f64 {
    let mut result = x + coeff[0];
    for &coefficient in &coeff[1..] {
        result = result * x + coefficient;
    }
    result
}

const ERF_T: [f64; 5] = [
    9.60497373987051638749e0, 9.00260197203842689217e1,
    2.23200534594684319226e3, 7.00332514112805075473e3,
    5.55923013010394962768e4,
];
const ERF_U: [f64; 5] = [
    3.35617141647503099647e1, 5.21357949780152679795e2,
    4.59432382970980127987e3, 2.26290000613890934246e4,
    4.92673942608635921086e4,
];

/// erfc(x) for finite nonnegative x; evaluated before casting to f32 so the
/// negative GELU tail is not lost by first rounding erf(x) to -1.0.
#[inline]
fn erfc_nonnegative_f64(x: f64) -> f64 {
    debug_assert!(x >= 0.0 && x.is_finite());
    if x < 1.0 {
        return 1.0 - x * horner(x * x, &ERF_T) / horner_leading_one(x * x, &ERF_U);
    }
    // exp(-27^2) is below the minimum positive f64 normal by several orders
    // and cannot contribute even an f32 subnormal GELU result.
    if x > 27.0 { return 0.0; }
    const P: [f64; 9] = [
        2.46196981473530512524e-10, 5.64189564831068821977e-1,
        7.46321056442269912687e0, 4.86371970985681366614e1,
        1.96520832956077098242e2, 5.26445194995477358631e2,
        9.34528527171957607540e2, 1.02755188689515710272e3,
        5.57535335369399327526e2,
    ];
    const Q: [f64; 8] = [
        1.32281951154744992508e1, 8.67072140885989742329e1,
        3.54937778887819891062e2, 9.75708501743205489753e2,
        1.82390916687909736289e3, 2.24633760818710981792e3,
        1.65666309194161350182e3, 5.57535340817727675546e2,
    ];
    const R: [f64; 6] = [
        5.64189583547755073984e-1, 1.27536670759978104416e0,
        5.01905042251180477414e0, 6.16021097993053585195e0,
        7.40974269950448939160e0, 2.97886665372100240670e0,
    ];
    const S: [f64; 6] = [
        2.26052863220117276590e0, 9.39603524938001434673e0,
        1.20489539808096656605e1, 1.70814450747565897222e1,
        9.60896809063285878198e0, 3.36907645100081516050e0,
    ];
    let fraction = if x < 8.0 {
        horner(x, &P) / horner_leading_one(x, &Q)
    } else {
        horner(x, &R) / horner_leading_one(x, &S)
    };
    (-x * x).exp() * fraction
}

#[inline]
fn erf_value_f32(x: f32) -> f32 {
    if x.is_nan() || x == 0.0 { return x; } // preserve NaN bits and +/-0
    if x.is_infinite() { return if x.is_sign_negative() { -1.0 } else { 1.0 }; }
    let ax = f64::from(x).abs();
    let value = if ax < 1.0 {
        // Avoid catastrophic cancellation for tiny positive subnormals:
        // 1 - erfc(x) would round to 0 even with a f64 intermediate.
        ax * horner(ax * ax, &ERF_T) / horner_leading_one(ax * ax, &ERF_U)
    } else {
        1.0 - erfc_nonnegative_f64(ax)
    };
    if x.is_sign_negative() { -(value as f32) } else { value as f32 }
}

#[inline]
fn gelu_erf_value_f32(x: f32) -> f32 {
    if x.is_nan() || x == 0.0 { return x; }
    if x == f32::NEG_INFINITY { return -0.0; }
    if x == f32::INFINITY { return f32::INFINITY; }
    let erfc = erfc_nonnegative_f64(f64::from(x).abs() / std::f64::consts::SQRT_2);
    let answer = if x.is_sign_negative() {
        0.5 * f64::from(x) * erfc
    } else {
        0.5 * f64::from(x) * (2.0 - erfc)
    };
    answer as f32
}

unary_f32!(erf_f32, x => erf_value_f32(x));
unary_f32!(gelu_erf_f32, x => gelu_erf_value_f32(x));

// Sigmoid uses algebraically equivalent, overflow-resistant branches. In
// particular, exp(-x) never overflows on very negative inputs, and the
// result tends to +0 and 1 for -infinity and +infinity respectively.
#[inline(always)]
fn sigmoid_value(x: f32) -> f32 {
    if x >= 0.0 {
        1.0_f32 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// log1p(exp(x)) with no overflow for large positive x and no cancellation
// for large negative x. A special NaN check is essential: f32::max normally
// selects the non-NaN operand, which would silently turn NaN into zero.
#[inline(always)]
fn softplus_value(x: f32) -> f32 {
    if x.is_nan() { return x; }
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

unary_f32!(sigmoid_f32, x => sigmoid_value(x));
// These are explicitly NaN-preserving. ReLU maps both signed zeros and
// negative values to +0; ReLU6 caps positive values at exactly 6.
unary_f32!(relu_f32, x => {
    if x.is_nan() { x } else if x > 0.0 { x } else { 0.0 }
});
unary_f32!(relu6_f32, x => {
    if x.is_nan() { x } else if x <= 0.0 { 0.0 } else if x >= 6.0 { 6.0 } else { x }
});
// x * sigmoid(x); handle the mathematically finite -infinity limit before
// multiplying -infinity by an underflowed zero.
unary_f32!(silu_f32, x => {
    if x == f32::NEG_INFINITY { -0.0 } else { x * sigmoid_value(x) }
});
unary_f32!(softplus_f32, x => softplus_value(x));
// A naive x / (1 + |x|) would produce NaN for either infinity.
unary_f32!(softsign_f32, x => {
    if x.is_infinite() { x.signum() } else { x / (1.0 + x.abs()) }
});
// x * clamp((x + 3) / 6, 0, 1) with safe +/-infinity endpoints.
unary_f32!(hard_swish_f32, x => {
    if x <= -3.0 { 0.0 } else if x >= 3.0 { x } else { x * ((x + 3.0) / 6.0) }
});
// x * tanh(softplus(x)), including a defined -infinity limiting case.
unary_f32!(mish_f32, x => {
    if x == f32::NEG_INFINITY { -0.0 } else { x * softplus_value(x).tanh() }
});
// ELU has no scalar argument in the canonical IR; it therefore fixes alpha=1.
// exp_m1 avoids loss of precision for small negative inputs. NaNs and signed
// zero are preserved without a branch to a relaxed-math implementation.
unary_f32!(elu_f32, x => {
    if x < 0.0 { x.exp_m1() } else { x }
});
// Canonical SELU parameters, frozen as IEEE-754 binary32 bit patterns.
// alpha=1.6732632423543772 -> 0x3fd62d7d;
// scale=1.0507009873554805 -> 0x3f867d5f.
// The multiplication order is fixed for reproducible strict-f32 evaluation.
unary_f32!(selu_f32, x => {
    let alpha = f32::from_bits(0x3fd6_2d7d);
    let scale = f32::from_bits(0x3f86_7d5f);
    if x < 0.0 { scale * (alpha * x.exp_m1()) } else { scale * x }
});
unary_f32!(floor_f32, x => x.floor());
unary_f32!(ceil_f32, x => x.ceil());
// Rust f32::round uses ties AWAY from zero, unlike bankers' rounding.
// Preserve IEEE signed zero, infinities, and NaN payload behavior of f32.
unary_f32!(round_f32, x => x.round());
// Preserve +/-0 and NaN without borrowing the f32::signum zero convention.
// Infinities and finite nonzero values collapse to +/-1 as in tensor Sign.
unary_f32!(sign_f32, x => {
    if x.is_nan() || x == 0.0 { x }
    else if x.is_sign_negative() { -1.0_f32 }
    else { 1.0_f32 }
});

/// The *explicitly approximate* GELU tanh formula. This is NOT an erf GELU
/// implementation and cannot bind either GeluErfExact or the ambiguous Gelu.
/// Saturated endpoints avoid -infinity * 0 becoming NaN. All intermediates
/// remain f32 under the selected strict-f32 arithmetic contract.
unary_f32!(gelu_tanh_approx_f32, x => {
    if x == f32::NEG_INFINITY { -0.0 }
    else if x == f32::INFINITY { f32::INFINITY }
    else {
        let cubic = x * x * x;
        let t = (0.797_884_6_f32 * (x + 0.044_715_f32 * cubic)).tanh();
        0.5_f32 * x * (1.0_f32 + t)
    }
});

/// Binary canonical min/max always canonicalize NaNs and signed zeros.
unsafe fn min_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe { *cmd.out.add(i) = crate::tensor_light::canonical_minimum(*cmd.a.add(i), *cmd.b.add(i)); }
    }
}
unsafe fn max_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len {
        unsafe { *cmd.out.add(i) = crate::tensor_light::canonical_maximum(*cmd.a.add(i), *cmd.b.add(i)); }
    }
}
unsafe fn neg_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len { unsafe { *cmd.out.add(i) = -*cmd.a.add(i); } }
}
unsafe fn abs_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len { unsafe { *cmd.out.add(i) = (*cmd.a.add(i)).abs(); } }
}
unsafe fn square_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len { unsafe { let x = *cmd.a.add(i); *cmd.out.add(i) = x*x; } }
}
unsafe fn sqrt_f32(cmd: &CpuCommand) {
    for i in 0..cmd.len { unsafe { *cmd.out.add(i) = (*cmd.a.add(i)).sqrt(); } }
}
/// A finite f32 intermediate is NOT evidence that a short reduction is
/// accurate: [1e8, 1, -1e8] loses the middle element and returns zero.
/// At most two terms can be safely folded with a single correctly rounded
/// f32 addition. Longer reductions use scalar f64 accumulation immediately,
/// avoiding both cancellation and the 2^24 addition plateau. Input and
/// output storage stay f32; no new buffer is allocated during replay.
unsafe fn sum_all_f32(cmd: &CpuCommand) {
    if cmd.k <= 2 {
        let mut sum32 = 0.0f32;
        for i in 0..cmd.k { sum32 += unsafe { *cmd.a.add(i) }; }
        if sum32.is_finite() {
            unsafe { *cmd.out = sum32; }
            return;
        }
    }
    let mut sum64 = 0.0f64;
    for i in 0..cmd.k { sum64 += f64::from(unsafe { *cmd.a.add(i) }); }
    unsafe { *cmd.out = sum64 as f32; }
}

/// The ordinary single-pass f32 product is kept whenever it produces a
/// finite, nonzero result (or contains a real zero): this is the lowest-cost
/// path. If a finite, all-nonzero product overflows or underflows midway,
/// replay the same inputs using bounded binary mantissa/exponent tracking.
/// This permits later factors to cancel a huge/tiny intermediate without
/// importing a different dtype or allocating any workspace. A f64 product
/// alone is not enough: hundreds of legal f32 factors can overflow f64 too.
unsafe fn product_all_f32(cmd: &CpuCommand) {
    let mut acc32 = 1.0f32;
    let mut real_zero = false;
    for i in 0..cmd.k {
        let value = unsafe { *cmd.a.add(i) };
        if value == 0.0 { real_zero = true; }
        acc32 *= value;
    }
    if acc32.is_finite() && (acc32 != 0.0 || real_zero) {
        unsafe { *cmd.out = acc32; }
        return;
    }

    let mut exponent = 0i128;
    let mut mantissa = 1.0f64;
    let mut negative = false;
    let mut has_zero = false;
    let mut has_infinity = false;
    for i in 0..cmd.k {
        let value = unsafe { *cmd.a.add(i) };
        if value.is_nan() {
            unsafe { *cmd.out = crate::tensor_light::CANONICAL_QUIET_NAN32; }
            return;
        }
        negative ^= value.is_sign_negative();
        if value == 0.0 { has_zero = true; continue; }
        if value.is_infinite() { has_infinity = true; continue; }
        // Every nonzero finite f32 converts to a NORMAL f64, including
        // f32 subnormals. Decode its binary exponent without logarithms.
        let bits = f64::from(value.abs()).to_bits();
        let power = ((bits >> 52) & 0x7ff) as i32 - 1023;
        let factor = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
        mantissa *= factor;
        exponent += i128::from(power);
        if mantissa >= 2.0 {
            mantissa *= 0.5;
            exponent += 1;
        }
    }
    // Invalid zero-times-infinity is always the canonical quiet NaN; do not
    // apply the product's sign parity to a NaN payload.
    if has_zero && has_infinity {
        unsafe { *cmd.out = crate::tensor_light::CANONICAL_QUIET_NAN32; }
        return;
    }
    let magnitude = if has_infinity { f32::INFINITY }
        else if has_zero { 0.0f32 }
        else if exponent >= 128 { f32::INFINITY }
        else if exponent < -150 { 0.0f32 }
        else { (mantissa * (2.0f64).powi(exponent as i32)) as f32 };
    unsafe { *cmd.out = if negative { -magnitude } else { magnitude }; }
}
unsafe fn count_nonzero_all_f32(cmd: &CpuCommand) {
    let mut count: u64 = 0;
    for i in 0..cmd.k { unsafe { if *cmd.a.add(i) != 0.0 { count += 1; } } }
    unsafe { *cmd.out = count as f32; }
}

// The scalar-only frozen family uses one direct, allocation-free pass per
// entry. `scalar0` and `scalar1` are bound once from exact FP32 bits.
macro_rules! scalar_kernel {
    ($name:ident, |$x:ident, $a:ident, $b:ident| $body:expr) => {
        unsafe fn $name(cmd: &CpuCommand) {
            let $a = cmd.scalar0;
            let $b = cmd.scalar1;
            for i in 0..cmd.len {
                let $x = unsafe { *cmd.a.add(i) };
                let value: f32 = $body;
                unsafe { *cmd.out.add(i) = value; }
            }
        }
    };
}
scalar_kernel!(scalar_add_f32, |x, a, _b| x + a);
scalar_kernel!(scalar_mul_f32, |x, a, _b| x * a);
scalar_kernel!(scalar_div_zero_to_zero_f32, |x, a, _b| {
    if a == 0.0 { 0.0 } else { x / a }
});
scalar_kernel!(scalar_leaky_relu_f32, |x, a, _b| {
    if x >= 0.0 { x } else { a * x }
});
scalar_kernel!(scalar_clamp_f32, |x, low, high| x.clamp(low, high));
scalar_kernel!(scalar_power_identity_f32, |x, _a, _b| x);
scalar_kernel!(scalar_power_square_f32, |x, _a, _b| x * x);
scalar_kernel!(scalar_power_safe_sqrt_f32, |x, _a, _b| {
    // Match the existing safe-sqrt operator: nonpositives map to +0, but
    // NaN continues through sqrt and remains NaN.
    if x <= 0.0 { 0.0 } else { x.sqrt() }
});
scalar_kernel!(scalar_power_safe_recip_f32, |x, _a, _b| {
    if x == 0.0 { 0.0 } else { 1.0 / x }
});
scalar_kernel!(scalar_power_three_halves_f32, |x, _a, _b| {
    if x > 0.0 { x * x.sqrt() } else { 0.0 }
});
scalar_kernel!(scalar_power_general_f32, |x, power, _b| {
    if x > 0.0 { (power * x.ln()).exp() } else { 0.0 }
});

/// A short mean can lose a small addend even WITHOUT overflow (for example
/// [1e8, 1, -1e8] / 3). A lone f32 input is already its exact mean;
/// otherwise accumulate into scalar f64 and divide before rounding to f32.
/// No tensor shape/dtype change or replay allocation is involved.
unsafe fn mean_all_f32(cmd: &CpuCommand) {
    if cmd.k == 0 {
        unsafe { *cmd.out = crate::tensor_light::CANONICAL_QUIET_NAN32; }
        return;
    }
    if cmd.k == 1 {
        unsafe { *cmd.out = *cmd.a; }
        return;
    }
    let mut sum64 = 0.0f64;
    for i in 0..cmd.k { sum64 += f64::from(unsafe { *cmd.a.add(i) }); }
    unsafe { *cmd.out = (sum64 / cmd.k as f64) as f32; }
}
/// Even three positive terms can lose a representable f32 result increment:
/// [2^24, 1, 1] has exact L1 norm 2^24+2, not 2^24. Retain the direct f32
/// path only where a single add is sufficient; otherwise accumulate as f64.
unsafe fn l1_norm_all_f32(cmd: &CpuCommand) {
    if cmd.k <= 2 {
        let mut sum32 = 0.0f32;
        for i in 0..cmd.k { sum32 += unsafe { (*cmd.a.add(i)).abs() }; }
        if sum32.is_finite() {
            unsafe { *cmd.out = sum32; }
            return;
        }
    }
    let mut sum64 = 0.0f64;
    for i in 0..cmd.k { sum64 += f64::from(unsafe { (*cmd.a.add(i)).abs() }); }
    unsafe { *cmd.out = sum64 as f32; }
}
/// The scaled-f32 recurrence may discard many small terms even in a SHORT
/// vector (2048 ones alongside 65536 suffice). Use the historical fast
/// scalar path for at most two inputs and a scalar f64 sum of squares
/// beyond that. Every f32 square and any addressable count fit in finite f64.
/// NaN takes precedence over infinity; tensor storage remains solely f32.
unsafe fn l2_norm_all_f32(cmd: &CpuCommand) {
    if cmd.k > 2 {
        let mut squares64 = 0.0f64;
        let mut has_nan = false;
        let mut has_inf = false;
        for i in 0..cmd.k {
            let x = unsafe { *cmd.a.add(i) };
            if x.is_nan() { has_nan = true; }
            else if x.is_infinite() { has_inf = true; }
            else {
                let wide = f64::from(x);
                squares64 += wide * wide;
            }
        }
        let result = if has_nan { f32::NAN }
            else if has_inf { f32::INFINITY }
            else { squares64.sqrt() as f32 };
        unsafe { *cmd.out = result; }
        return;
    }
    let mut scale = 0.0_f32;
    let mut sumsq = 1.0_f32;
    let mut has_nan = false;
    let mut has_inf = false;
    for i in 0..cmd.k {
        let x = unsafe { (*cmd.a.add(i)).abs() };
        if x.is_nan() { has_nan = true; }
        else if x.is_infinite() { has_inf = true; }
        else if x != 0.0 {
            if scale < x {
                let ratio = scale / x;
                sumsq = 1.0 + sumsq * ratio * ratio;
                scale = x;
            } else {
                let ratio = x / scale;
                sumsq += ratio * ratio;
            }
        }
    }
    let result = if has_nan { f32::NAN }
        else if has_inf { f32::INFINITY }
        else if scale == 0.0 { 0.0 }
        else { scale * sumsq.sqrt() };
    unsafe { *cmd.out = result; }
}
unsafe fn linf_norm_all_f32(cmd: &CpuCommand) {
    let mut max_abs = 0.0_f32;
    for i in 0..cmd.k {
        let value = unsafe { (*cmd.a.add(i)).abs() };
        if value.is_nan() {
            unsafe { *cmd.out = f32::NAN; }
            return;
        }
        max_abs = max_abs.max(value);
    }
    unsafe { *cmd.out = max_abs; }
}

/// One-pass population Welford moments with SCALAR f64 intermediates.
/// Unlike normalizing by a large f32 magnitude, this does not lose real
/// variance between adjacent representable f32 inputs. For example, the
/// variance of [100_000_000, 100_000_008] is 16, not zero. F32 magnitude
/// squared fits easily in f64; no two-pass scale scan or temporary storage is
/// necessary. Output dtype, shape, axis selection and f32 storage are fixed.
/// Nonfinite samples yield canonical NaN and empty moments are rejected at BIND.
unsafe fn population_moment_f32(cmd: &CpuCommand, stddev: bool) {
    if cmd.k == 0 {
        unsafe { *cmd.out = crate::tensor_light::CANONICAL_QUIET_NAN32; }
        return;
    }
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for i in 0..cmd.k {
        let sample = unsafe { *cmd.a.add(i) };
        if !sample.is_finite() {
            unsafe { *cmd.out = crate::tensor_light::CANONICAL_QUIET_NAN32; }
            return;
        }
        let x = f64::from(sample);
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (x - mean);
    }
    // Clamp only harmless negative round-off: true population variance is >=0.
    let variance = (m2 / cmd.k as f64).max(0.0);
    let result = if stddev { variance.sqrt() } else { variance };
    unsafe { *cmd.out = result as f32; }
}
unsafe fn variance_population_all_f32(cmd: &CpuCommand) {
    unsafe { population_moment_f32(cmd, false); }
}
unsafe fn stddev_population_all_f32(cmd: &CpuCommand) {
    unsafe { population_moment_f32(cmd, true); }
}

/// Two-pass max-shifted LogSumExp. NaNs poison the reduction, an all
/// -infinity input returns -infinity and +infinity dominates finite inputs.
/// The short f32 accumulation keeps its original order and cost. Long axes
/// use a scalar f64 sum of f32 exponential weights: above 2^24 identical
/// weights an f32 running sum can stall, producing an error of ln(2) or more.
/// The Tensor input and scalar result remain exclusively f32.
unsafe fn logsumexp_all_f32(cmd: &CpuCommand) {
    let mut maximum = f32::NEG_INFINITY;
    for i in 0..cmd.k {
        let value = unsafe { *cmd.a.add(i) };
        if value.is_nan() {
            unsafe { *cmd.out = f32::NAN; }
            return;
        }
        maximum = maximum.max(value);
    }
    if maximum.is_infinite() {
        unsafe { *cmd.out = maximum; }
        return;
    }
    if cmd.k >= 4096 {
        let mut total64 = 0.0f64;
        for i in 0..cmd.k {
            total64 += f64::from((unsafe { *cmd.a.add(i) } - maximum).exp());
        }
        unsafe { *cmd.out = maximum + total64.ln() as f32; }
    } else {
        let mut total32 = 0.0f32;
        for i in 0..cmd.k {
            total32 += (unsafe { *cmd.a.add(i) } - maximum).exp();
        }
        unsafe { *cmd.out = maximum + total32.ln(); }
    }
}

/// Canonical all-axis min/max: propagate NaNs with the canonical bit pattern
/// and order -0 below +0. Empty reductions are NOT admissible without an
/// explicitly defined empty identity, so bind rejects k=0 for these entries.
unsafe fn minimum_all_f32(cmd: &CpuCommand) {
    debug_assert!(cmd.k > 0);
    let first = unsafe { *cmd.a };
    let mut best = crate::tensor_light::canonical_minimum(first, first);
    for i in 1..cmd.k {
        best = crate::tensor_light::canonical_minimum(best, unsafe { *cmd.a.add(i) });
    }
    unsafe { *cmd.out = best; }
}
unsafe fn maximum_all_f32(cmd: &CpuCommand) {
    debug_assert!(cmd.k > 0);
    let first = unsafe { *cmd.a };
    let mut best = crate::tensor_light::canonical_maximum(first, first);
    for i in 1..cmd.k {
        best = crate::tensor_light::canonical_maximum(best, unsafe { *cmd.a.add(i) });
    }
    unsafe { *cmd.out = best; }
}

/// Translate one element of a GroupNorm channel partition to a global
/// contiguous f32 offset. The batch axis is 0 and only the channel coordinate
/// is shifted by the frozen group index. Bind checked all products and
/// dimensions; replay never allocates or reinterprets the dtype.
#[inline]
fn group_channel_offset(
    position: usize, group: usize, axes: ArbitraryAxes, group_count: usize,
) -> usize {
    let channel_axis = axes.mask.trailing_zeros() as usize;
    let channels_per_group = axes.dims[channel_axis] / group_count;
    let batch = group / group_count;
    let channel_group = group % group_count;
    let mut linear = position;
    let mut stride = 1usize;
    let mut offset = 0usize;
    for axis in (1..usize::from(axes.rank)).rev() {
        let local_extent = if axis == channel_axis {
            channels_per_group
        } else { axes.dims[axis] };
        let coordinate = linear % local_extent;
        linear /= local_extent;
        let coordinate = if axis == channel_axis {
            channel_group * channels_per_group + coordinate
        } else { coordinate };
        offset += coordinate * stride;
        stride *= axes.dims[axis];
    }
    debug_assert_eq!(linear, 0);
    offset + batch * stride
}

/// Population GroupNorm for a non-leading physical channel axis.
/// Contiguous-channel cases bind the faster normalize_suffix_f32::<0>.
unsafe fn normalize_group_channels_f32(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let axes = cmd.arbitrary_axes.expect("group geometry bound at BIND");
    let epsilon = f64::from(cmd.scalar0);
    for group in 0..cmd.m {
        let mut sum = 0.0f64;
        let mut finite = true;
        for position in 0..cmd.k {
            let offset = group_channel_offset(position, group, axes, cmd.n);
            let value = unsafe { *cmd.a.add(offset) };
            if !value.is_finite() { finite = false; break; }
            sum += f64::from(value);
        }
        if !finite {
            for position in 0..cmd.k {
                let offset = group_channel_offset(position, group, axes, cmd.n);
                unsafe { *cmd.out.add(offset) = crate::tensor_light::CANONICAL_QUIET_NAN32; }
            }
            continue;
        }
        let mean = sum / cmd.k as f64;
        let mut sum_of_squares = 0.0f64;
        for position in 0..cmd.k {
            let offset = group_channel_offset(position, group, axes, cmd.n);
            let diff = f64::from(unsafe { *cmd.a.add(offset) }) - mean;
            sum_of_squares += diff * diff;
        }
        let denominator = (sum_of_squares / cmd.k as f64 + epsilon).sqrt();
        for position in 0..cmd.k {
            let offset = group_channel_offset(position, group, axes, cmd.n);
            let value = f64::from(unsafe { *cmd.a.add(offset) });
            unsafe { *cmd.out.add(offset) = ((value - mean) / denominator) as f32; }
        }
    }
}

/// Strict FP32 reduction order along the K axis for each output element.
/// No FMA/TF32/fast math may be enabled by the selected artifact.
/// Stable, allocation-free and exact-shape suffix normalization. The chosen
/// entry contains a compile-time normalization kind (0=LayerNorm, 1=RMSNorm,
/// 2=L2Norm); `scalar0` is an immutable, prevalidated f32 epsilon. Only scalar
/// accumulators use f64 for numerical stability, not any Tensor or ABI output.
/// Nonfinite input poisons its *own* normalization group with canonical NaNs.
unsafe fn normalize_suffix_f32<const KIND: u8>(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let count = cmd.k;
    let epsilon = cmd.scalar0 as f64;
    for group in 0..cmd.m {
        let base = group * count;
        let mut sum = 0.0_f64;
        let mut sum_of_squares = 0.0_f64;
        let mut has_nonfinite = false;
        for position in 0..count {
            let value = unsafe { *cmd.a.add(base + position) };
            if !value.is_finite() {
                has_nonfinite = true;
                break;
            }
            let x = f64::from(value);
            if KIND == 0 { sum += x; }
            else { sum_of_squares += x * x; }
        }
        if has_nonfinite {
            for position in 0..count {
                unsafe { *cmd.out.add(base + position) = crate::tensor_light::CANONICAL_QUIET_NAN32; }
            }
            continue;
        }
        let mean = if KIND == 0 { sum / count as f64 } else { 0.0 };
        if KIND == 0 {
            for position in 0..count {
                let x = f64::from(unsafe { *cmd.a.add(base + position) }) - mean;
                sum_of_squares += x * x;
            }
        }
        let scale = if KIND == 2 { sum_of_squares + epsilon }
                    else { sum_of_squares / count as f64 + epsilon };
        let denominator = scale.sqrt();
        for position in 0..count {
            let x = f64::from(unsafe { *cmd.a.add(base + position) });
            let numerator = if KIND == 0 { x - mean } else { x };
            unsafe { *cmd.out.add(base + position) = (numerator / denominator) as f32; }
        }
    }
}

/// Convert an index within the selected/nonselected axes into an offset
/// in the full contiguous tensor. All dimensions and products were proved at
/// BIND, and empty tensors return before this routine is ever called.
#[inline]
fn arbitrary_axis_offset(mut index: usize, axes: ArbitraryAxes, selected: bool) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for axis in (0..usize::from(axes.rank)).rev() {
        let dimension = axes.dims[axis];
        if ((axes.mask & (1_u8 << axis)) != 0) == selected {
            let coordinate = index % dimension;
            index /= dimension;
            offset += coordinate * stride;
        }
        stride *= dimension;
    }
    debug_assert_eq!(index, 0);
    offset
}

/// O(rank * numel), allocation-free generic-axis fallback for LayerNorm,
/// RMSNorm and L2Norm; suffix-normalization uses the faster direct kernel.
/// Input/output buffers are validated disjoint and shape-equal at BIND.
unsafe fn normalize_arbitrary_axes_f32<const KIND: u8>(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let axes = cmd.arbitrary_axes.expect("generic axis metadata must be bound");
    let epsilon = f64::from(cmd.scalar0);
    for group in 0..cmd.m {
        let group_base = arbitrary_axis_offset(group, axes, false);
        let mut sum = 0.0_f64;
        let mut sum_of_squares = 0.0_f64;
        let mut finite = true;
        for position in 0..cmd.k {
            let offset = group_base + arbitrary_axis_offset(position, axes, true);
            let value = unsafe { *cmd.a.add(offset) };
            if !value.is_finite() { finite = false; break; }
            let value = f64::from(value);
            if KIND == 0 { sum += value; }
            else { sum_of_squares += value * value; }
        }
        if !finite {
            for position in 0..cmd.k {
                let offset = group_base + arbitrary_axis_offset(position, axes, true);
                unsafe { *cmd.out.add(offset) = crate::tensor_light::CANONICAL_QUIET_NAN32; }
            }
            continue;
        }
        let mean = if KIND == 0 { sum / cmd.k as f64 } else { 0.0 };
        if KIND == 0 {
            for position in 0..cmd.k {
                let offset = group_base + arbitrary_axis_offset(position, axes, true);
                let x = f64::from(unsafe { *cmd.a.add(offset) }) - mean;
                sum_of_squares += x * x;
            }
        }
        let scale = if KIND == 2 { sum_of_squares + epsilon }
                    else { sum_of_squares / cmd.k as f64 + epsilon };
        let denom = scale.sqrt();
        for position in 0..cmd.k {
            let offset = group_base + arbitrary_axis_offset(position, axes, true);
            let x = f64::from(unsafe { *cmd.a.add(offset) });
            let numerator = if KIND == 0 { x - mean } else { x };
            unsafe { *cmd.out.add(offset) = (numerator / denom) as f32; }
        }
    }
}

/// Stable, allocation-free softmax over any one contiguous logical axis.
/// m/k/n are bound once; k is nonzero. NaNs poison the complete slice.
/// An all -infinity slice produces canonical NaNs. Positive infinities share
/// probability mass equally (log-softmax reports -ln(count) at those positions).
/// This policy is deliberate: naive (inf - inf) would otherwise create NaNs.
unsafe fn softmax_axis_impl<const LOG: bool>(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let row_stride = cmd.k * cmd.n;
    for outer in 0..cmd.m {
        for inner in 0..cmd.n {
            let base = outer * row_stride + inner;
            let mut maximum = f32::NEG_INFINITY;
            let mut nan_found = false;
            for position in 0..cmd.k {
                let value = unsafe { *cmd.a.add(base + position * cmd.n) };
                if value.is_nan() { nan_found = true; }
                if value > maximum { maximum = value; }
            }
            if nan_found || maximum == f32::NEG_INFINITY {
                for position in 0..cmd.k {
                    unsafe { *cmd.out.add(base + position * cmd.n) = crate::tensor_light::CANONICAL_QUIET_NAN32; }
                }
                continue;
            }
            if maximum == f32::INFINITY {
                let mut count = 0usize;
                for position in 0..cmd.k {
                    if unsafe { *cmd.a.add(base + position * cmd.n) } == f32::INFINITY {
                        count += 1;
                    }
                }
                let positive = if LOG { -(count as f32).ln() } else { 1.0 / count as f32 };
                let other = if LOG { f32::NEG_INFINITY } else { 0.0 };
                for position in 0..cmd.k {
                    let value = unsafe { *cmd.a.add(base + position * cmd.n) };
                    unsafe { *cmd.out.add(base + position * cmd.n) =
                        if value == f32::INFINITY { positive } else { other }; }
                }
                continue;
            }
            // One exp per element, stored in the disjoint bound output for
            // ordinary Softmax; LogSoftmax needs no temporary output writes.
            // Keep the inexpensive original f32 reduction/division for short
            // axes. For long axes, repeated f32 additions can STOP changing
            // once the partial sum reaches 2^24 (e.g. >2^24 equal logits),
            // causing probabilities to sum to ~2 instead of 1. A scalar f64
            // accumulator prevents this loss of significant contributions.
            // The public input/output buffers are still exclusively f32.
            // Choosing the path once per row introduces no allocation and
            // does not change the short-axis exact-bit behavior.
            let wide_sum = cmd.k >= 4096;
            let mut total32 = 0.0f32;
            let mut total64 = 0.0f64;
            for position in 0..cmd.k {
                let offset = base + position * cmd.n;
                let exponential = (unsafe { *cmd.a.add(offset) } - maximum).exp();
                if wide_sum { total64 += f64::from(exponential); }
                else { total32 += exponential; }
                if !LOG { unsafe { *cmd.out.add(offset) = exponential; } }
            }
            if LOG {
                let normalization = if wide_sum { total64.ln() as f32 }
                                    else { total32.ln() };
                for position in 0..cmd.k {
                    let offset = base + position * cmd.n;
                    let x = unsafe { *cmd.a.add(offset) };
                    unsafe { *cmd.out.add(offset) = x - maximum - normalization; }
                }
            } else if wide_sum {
                // One reciprocal replaces an elementwise division on the long
                // axis. This is a declared f32 output rounding step, not a
                // relaxed-math/TF32 conversion of the input Tensor.
                let inverse = (1.0f64 / total64) as f32;
                for position in 0..cmd.k {
                    let offset = base + position * cmd.n;
                    unsafe { *cmd.out.add(offset) *= inverse; }
                }
            } else {
                for position in 0..cmd.k {
                    let offset = base + position * cmd.n;
                    unsafe { *cmd.out.add(offset) /= total32; }
                }
            }
        }
    }
}

unsafe fn softmax_axis_f32(cmd: &CpuCommand) {
    unsafe { softmax_axis_impl::<false>(cmd); }
}
unsafe fn log_softmax_axis_f32(cmd: &CpuCommand) {
    unsafe { softmax_axis_impl::<true>(cmd); }
}

/// Materialize a flip of a single axis or a consecutive block on an
/// independently bound contiguous f32 output. Original bits (including signed
/// zero and NaN payloads) are copied unchanged. For nonempty tensors the
/// (outer, flattened_block, inner) geometry was checked at BIND. Bulk inner-axis
/// chunks are copied in one contiguous operation; replay allocates nothing.
unsafe fn flip_axis_f32(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let row_stride = cmd.k * cmd.n;
    for outer in 0..cmd.m {
        let row_base = outer * row_stride;
        for position in 0..cmd.k {
            let source = row_base + (cmd.k - 1 - position) * cmd.n;
            let destination = row_base + position * cmd.n;
            // SAFETY: bind checks total size, exact geometry and disjoint
            // physical inputs/outputs. Each chunk is within its owner span.
            unsafe {
                ptr::copy_nonoverlapping(
                    cmd.a.add(source), cmd.out.add(destination), cmd.n,
                );
            }
        }
    }
}

/// A general f32 reversal of ANY distinct subset of rank-1..8 axes.
/// The fast flattened-block variant remains preferred where applicable.
/// Reverse the logical coordinates without changing rank, dims, or any f32
/// payload bits (including signed zero and noncanonical NaN payloads).
unsafe fn flip_arbitrary_axes_f32(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let axes = cmd.arbitrary_axes.expect("generic axis metadata must be bound");
    for destination in 0..cmd.len {
        let mut coordinates = destination;
        let mut stride = 1usize;
        let mut source = 0usize;
        for axis in (0..usize::from(axes.rank)).rev() {
            let dimension = axes.dims[axis];
            let coordinate = coordinates % dimension;
            coordinates /= dimension;
            let mapped = if (axes.mask & (1_u8 << axis)) != 0 {
                dimension - 1 - coordinate
            } else { coordinate };
            source += mapped * stride;
            stride *= dimension;
        }
        debug_assert_eq!(coordinates, 0);
        unsafe {
            ptr::copy_nonoverlapping(cmd.a.add(source), cmd.out.add(destination), 1);
        }
    }
}

/// Stable ascending/descending sort of raw f32 VALUES for any selected
/// contiguous-storage axis. The output has precisely the input's shape and
/// dtype. The only workspace is fixed at BIND; std's sort_unstable_by is
/// in-place and does not allocate during replay. The original ordinal is a
/// secondary comparison key so equal (including +/-0) values and all NaNs
/// retain their source order. NaN placement is independent of sort direction.
unsafe fn sort_values_axis_f32<const DESCENDING: bool, const NANS_FIRST: bool>(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    // The binder allocated at least k items and retains the Vec allocation.
    // The bound backend executes its commands serially through &mut self.
    let scratch = unsafe { core::slice::from_raw_parts_mut(cmd.sort_scratch, cmd.k) };
    let row_stride = cmd.k * cmd.n;
    for outer in 0..cmd.m {
        for inner in 0..cmd.n {
            let base = outer * row_stride + inner;
            for position in 0..cmd.k {
                scratch[position] = SortRecord {
                    bits: unsafe { (*cmd.a.add(base + position * cmd.n)).to_bits() },
                    ordinal: position,
                };
            }
            scratch.sort_unstable_by(|left, right| {
                let a = f32::from_bits(left.bits);
                let b = f32::from_bits(right.bits);
                let ordering = match (a.is_nan(), b.is_nan()) {
                    (true, true) => core::cmp::Ordering::Equal,
                    (true, false) => if NANS_FIRST {
                        core::cmp::Ordering::Less
                    } else { core::cmp::Ordering::Greater },
                    (false, true) => if NANS_FIRST {
                        core::cmp::Ordering::Greater
                    } else { core::cmp::Ordering::Less },
                    (false, false) => {
                        let result = a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal);
                        if DESCENDING { result.reverse() } else { result }
                    },
                };
                ordering.then_with(|| left.ordinal.cmp(&right.ordinal))
            });
            for position in 0..cmd.k {
                unsafe { *cmd.out.add(base + position * cmd.n) =
                    f32::from_bits(scratch[position].bits); }
            }
        }
    }
}

/// Numerically stable two-term logaddexp. No double-precision Tensor is made.
#[inline]
fn scan_logaddexp(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() { return crate::tensor_light::CANONICAL_QUIET_NAN32; }
    if left == f32::NEG_INFINITY { return right; }
    if right == f32::NEG_INFINITY { return left; }
    if left == f32::INFINITY || right == f32::INFINITY { return f32::INFINITY; }
    let high = left.max(right);
    high + (-(left - right).abs()).exp().ln_1p()
}

/// Axis-wise f32 scan. KIND is frozen in the native entry, not selected inside
/// a hot dynamic dispatch. 0=sum, 1=product, 2=min, 3=max, 4=logsumexp.
/// Exclusive prefix is admissible only for KIND=0 (the canonical IR rule).
unsafe fn scan_axis_f32<const KIND: u8, const INCLUSIVE: bool>(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    let row_stride = cmd.k * cmd.n;
    for outer in 0..cmd.m {
        for inner in 0..cmd.n {
            let base = outer * row_stride + inner;
            let mut accumulator = if KIND == 0 { 0.0_f32 }
                else if KIND == 1 { 1.0_f32 }
                else if KIND == 4 || KIND == 3 { f32::NEG_INFINITY }
                else { f32::INFINITY };
            for position in 0..cmd.k {
                let offset = base + position * cmd.n;
                let value = unsafe { *cmd.a.add(offset) };
                if !INCLUSIVE { unsafe { *cmd.out.add(offset) = accumulator; } }
                accumulator = if position == 0 && INCLUSIVE {
                    if KIND == 2 { crate::tensor_light::canonical_minimum(value, value) }
                    else if KIND == 3 { crate::tensor_light::canonical_maximum(value, value) }
                    else { value }
                } else if KIND == 0 { accumulator + value }
                else if KIND == 1 { accumulator * value }
                else if KIND == 2 { crate::tensor_light::canonical_minimum(accumulator, value) }
                else if KIND == 3 { crate::tensor_light::canonical_maximum(accumulator, value) }
                else { scan_logaddexp(accumulator, value) };
                // Preserve the project's canonical NaN-result contract even
                // for a first-prefix source NaN with a noncanonical payload,
                // or for a later inf+(-inf) invalid arithmetic result.
                if accumulator.is_nan() { accumulator = crate::tensor_light::CANONICAL_QUIET_NAN32; }
                if INCLUSIVE { unsafe { *cmd.out.add(offset) = accumulator; } }
            }
        }
    }
}

unsafe fn gemm_dot_f32(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    for row in 0..cmd.m {
        for col in 0..cmd.n {
            let mut sum = 0.0_f32;
            for p in 0..cmd.k {
                let ai = if cmd.lhs_vector { p } else if cmd.transpose_lhs {
                    p * cmd.m + row
                } else { row * cmd.k + p };
                let bi = if cmd.rhs_vector { p } else if cmd.transpose_rhs {
                    col * cmd.k + p
                } else { p * cmd.n + col };
                // SAFETY: compatible rank-1/2 dimensions and C-order spans
                // were checked before producing this immutable command.
                unsafe { sum += *cmd.a.add(ai) * *cmd.b.add(bi); }
            }
            unsafe { *cmd.out.add(row * cmd.n + col) = sum; }
        }
    }
}

/// AXPY-style outer accumulation improves contiguous-B locality. Reduction
/// order along K is unchanged (0..K), with no approximate math.
unsafe fn gemm_axpy_f32(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    for i in 0..cmd.m * cmd.n {
        unsafe { *cmd.out.add(i) = 0.0; }
    }
    for row in 0..cmd.m {
        for p in 0..cmd.k {
            let a = unsafe { *cmd.a.add(row * cmd.k + p) };
            for col in 0..cmd.n {
                let c = unsafe { cmd.out.add(row * cmd.n + col) };
                unsafe { *c += a * *cmd.b.add(p * cmd.n + col); }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(cmd: &CpuCommand) { unsafe { binary_avx2::<0>(cmd) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_avx2(cmd: &CpuCommand) { unsafe { binary_avx2::<1>(cmd) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(cmd: &CpuCommand) { unsafe { binary_avx2::<2>(cmd) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn div_avx2(cmd: &CpuCommand) { unsafe { binary_avx2::<3>(cmd) } }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn binary_avx2<const OP: u8>(cmd: &CpuCommand) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    while cmd.len - i >= 8 {
        let (x,y) = unsafe { (_mm256_loadu_ps(cmd.a.add(i)), _mm256_loadu_ps(cmd.b.add(i))) };
        let v = if OP == 0 { _mm256_add_ps(x,y) }
            else if OP == 1 { _mm256_sub_ps(x,y) }
            else if OP == 2 { _mm256_mul_ps(x,y) }
            else { _mm256_div_ps(x,y) };
        unsafe { _mm256_storeu_ps(cmd.out.add(i),v); }
        i += 8;
    }
    while i < cmd.len {
        unsafe {
            let a = *cmd.a.add(i);
            let b = *cmd.b.add(i);
            *cmd.out.add(i) = if OP == 0 { a+b } else if OP == 1 { a-b }
                else if OP == 2 { a*b } else { a/b };
        }
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemm_avx2(cmd: &CpuCommand) {
    if cmd.len == 0 { return; }
    use std::arch::x86_64::*;
    // Keep one 8-lane accumulator in a register for the complete K reduction.
    // The previous implementation loaded and stored each output vector for
    // EVERY K iteration. This form stores C once while preserving the same
    // increasing-K arithmetic order and avoiding FMA/precision changes.
    for row in 0..cmd.m {
        let mut col = 0usize;
        while cmd.n - col >= 8 {
            let mut acc = _mm256_setzero_ps();
            for p in 0..cmd.k {
                let a = _mm256_set1_ps(unsafe { *cmd.a.add(row * cmd.k + p) });
                let b = unsafe { _mm256_loadu_ps(cmd.b.add(p * cmd.n + col)) };
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
            }
            unsafe { _mm256_storeu_ps(cmd.out.add(row * cmd.n + col), acc); }
            col += 8;
        }
        // Odd column tails use the same scalar reduction order as the strict
        // portable GEMM; K=0 explicitly writes the +0 identity.
        while col < cmd.n {
            let mut acc = 0.0_f32;
            for p in 0..cmd.k {
                unsafe { acc += *cmd.a.add(row * cmd.k + p) * *cmd.b.add(p * cmd.n + col); }
            }
            unsafe { *cmd.out.add(row * cmd.n + col) = acc; }
            col += 1;
        }
    }
}

#[inline(always)]
fn add_entry_avx2() -> unsafe fn(&CpuCommand) {
    #[cfg(target_arch = "x86_64")]
    { add_avx2 }
    #[cfg(not(target_arch = "x86_64"))]
    { add_f32 }
}
#[inline(always)]
fn sub_entry_avx2() -> unsafe fn(&CpuCommand) {
    #[cfg(target_arch = "x86_64")]
    { sub_avx2 }
    #[cfg(not(target_arch = "x86_64"))]
    { sub_f32 }
}
#[inline(always)]
fn mul_entry_avx2() -> unsafe fn(&CpuCommand) {
    #[cfg(target_arch = "x86_64")]
    { mul_avx2 }
    #[cfg(not(target_arch = "x86_64"))]
    { mul_f32 }
}
#[inline(always)]
fn div_entry_avx2() -> unsafe fn(&CpuCommand) {
    #[cfg(target_arch = "x86_64")]
    { div_avx2 }
    #[cfg(not(target_arch = "x86_64"))]
    { div_f32 }
}
#[inline(always)]
fn gemm_entry_avx2() -> unsafe fn(&CpuCommand) {
    #[cfg(target_arch = "x86_64")]
    { gemm_avx2 }
    #[cfg(not(target_arch = "x86_64"))]
    { gemm_dot_f32 }
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/cpu_native_tests.rs"]
mod cpu_native_tests;
