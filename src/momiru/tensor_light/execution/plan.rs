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
// Subcomponent:  Plan
// Module:        momiru::tensor_light::execution::plan
// Source:        src/tensor_light/execution/plan.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Single immutable UASDEA execution-plan model for resources, placements,
//   synchronization and deterministic identity.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Represent CPU, GPU and hybrid placements through the same plan schema.
//   - Specify static resource, synchronization and target binding requirements.
//   - Retain frozen schedule and candidate evidence required for plan identity.
//   - Do not perform runtime compilation or online autotuning.
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

//! Immutable execution-plan representation.
//!
//! CPU-only, GPU-only, and CPU-GPU execution are placement mappings of this
//! single plan model. The module also owns static resources, synchronization,
//! target IO bindings, candidate evidence, and deterministic plan identity.

use core::hash::{Hash, Hasher};

use super::ir::{
    CanonicalWorkload, KernelMathMode, NodeId, Operation, PlanKey, TensorId, TensorLayout,
    TensorSpec, WorkloadSignature,
};
use super::target::{ArtifactRef, FeatureMask, HardwareContract, LibraryAlgorithmPlan};
#[cfg(feature = "aot")]
use super::PlanError;
use super::{stable_hash_pair, StableIdentityHasher};
use super::{
    ArenaId, BarrierId, EventId, ExecutionContextId, MemorySlotId, StorageId, StreamId, SyncId,
    TransferBatchId, TransferId,
};

// =============================================================================
// Placement and kernel mapping
// =============================================================================

// =============================================================================
// CPU mapping: static parallelism, SIMD, cache/NUMA and prefetch choices
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrefetchKind {
    None,
    ReadL1,
    ReadL2,
    ReadL3,
    /// Read-shared prefetch candidate (for example x86 MOVRS/prefetchrs when available).
    ReadShared,
    WriteL1,
    NonTemporal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PagePolicy {
    BasePages,
    TransparentHugePages,
    ExplicitHugePages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StorePolicy {
    Cached,
    NonTemporal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WorkerWaitPolicy {
    Spin,
    SpinThenPark,
    Park,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NumaPolicy {
    Local,
    Fixed,
    Interleave,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuParallelKind {
    SingleThread,
    StaticShard,
    Static2D,
    Static3D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuDispatchMode {
    /// Execute directly on the replay caller thread.
    CallerThread,
    /// Workers are created, affinitized and kept alive during one-time binding.
    /// Replay only posts the frozen command; it never creates/tears down threads.
    PersistentWorkerTeam,
}

/// How the selected CPU AOT artifact was authored. The choice is made offline;
/// replay only calls the already-selected entry point. Portable SIMD is not used
/// as a required baseline unless the selected toolchain supports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuKernelSource {
    AutoVectorizedRust,
    StdArchIntrinsics,
    /// Nightly-only portable SIMD may be benchmarked offline, but is never a required baseline.
    PortableSimdNightly,
    /// Nightly-only core intrinsics when a verified artifact beats stable alternatives.
    CoreIntrinsicsNightly,
    InlineAsm,
    GlobalAsm,
    NakedAsm,
    VendorLibrary,
    ExternalNativeObject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LtoMode {
    Off,
    Thin,
    Fat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PgoMode {
    None,
    InstrumentedProfileUse,
    /// Kept as an offline/nightly artifact option when the selected toolchain supports it.
    SampleProfileUse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RustFloatMode {
    /// Ordinary Rust floating-point semantics.
    Strict,
    /// `algebraic_*`-style operations when supported by the selected toolchain and allowed by the accuracy contract.
    Algebraic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RustPanicStrategy {
    /// Never accepted for a maximum-throughput frozen artifact.
    Unwind,
    Abort,
    /// Cargo/rustc unstable option; may win offline for the smallest/lowest-overhead artifact.
    ImmediateAbort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PostLinkOptimization {
    None,
    /// Offline profile-guided binary layout/optimization, if the selected toolchain supports it.
    ProfileGuided,
}

/// Frozen metadata for the Rust/native compilation of a CPU artifact. These
/// settings are not selected in replay; the artifact hash identifies the exact
/// binary produced with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RustCodegenPlan {
    pub source: CpuKernelSource,
    pub rustc_version: &'static str,
    pub target_triple: &'static str,
    pub target_cpu: &'static str,
    pub target_features: &'static str,
    /// Optional `-Z tune-cpu` value for a nightly-produced artifact; empty means target_cpu/default.
    pub tune_cpu: &'static str,
    pub nightly_codegen: bool,
    pub opt_level: u8,
    pub lto: LtoMode,
    /// Cross-language/plugin LTO may be useful for mixed Rust/native artifacts.
    pub linker_plugin_lto: bool,
    pub codegen_units: u16,
    /// Empty string means compiler default. Otherwise these freeze rustc/LLVM code-model choices.
    pub code_model: &'static str,
    pub relocation_model: &'static str,
    pub vectorize_loops: bool,
    pub vectorize_slp: bool,
    /// Per-artifact Rust floating-point semantics; Algebraic maps to the selected toolchain's algebraic floating-point mode.
    pub float_mode: RustFloatMode,
    pub pgo: PgoMode,
    pub pgo_profile_hash_hi: u64,
    pub pgo_profile_hash_lo: u64,
    pub post_link_optimization: PostLinkOptimization,
    pub post_link_profile_hash_hi: u64,
    pub post_link_profile_hash_lo: u64,
    pub panic_strategy: RustPanicStrategy,
    /// Frozen `-C force-frame-pointers` choice; benchmarked offline with the artifact.
    pub force_frame_pointers: bool,
    pub overflow_checks: bool,
    /// Release artifacts must not retain debug-assertion branches in hot code.
    pub debug_assertions: bool,
    /// Incremental compilation is disabled for the frozen release artifact.
    pub incremental: bool,
    pub cold_error_paths: bool,
    /// Non-zero when non-default LLVM/codegen arguments affected the selected artifact.
    pub llvm_args_hash_hi: u64,
    pub llvm_args_hash_lo: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KernelFamily {
    /// Generic exact artifact for an operation without a more specific family tag.
    Generic,
    /// One artifact implementing multiple canonical semantic nodes as one frozen fusion.
    FusedStaticDataflow,
    ContiguousCopy,
    StridedCopy,
    /// Metadata-only zero-copy view. Correctness requires exact input/output
    /// alias-layout validation. Binding resolves the alias and replay emits no
    /// CPU call or GPU launch for this family.
    AliasView,
    ExpandMaterialize,
    FlipContiguous2d,
    FlipGeneric,
    Concat2,
    Concat4,
    ConcatGeneric,
    Stack2,
    Stack4,
    StackGeneric,
    PadConstantContiguous,
    PadMapped,
    ResampleNearest,
    ResampleBilinear,
    DownsampleMeanScale2,
    DownsampleMeanGeneric,
    DownsampleMeanIntegral2d,
    SlidingWindowContiguous,
    /// Rank-1 zero-insertion dilation with factor embedded in the AOT artifact.
    ZeroInsert1d,
    /// Periodic materialization with modulo source indexing.
    TileMaterialize,
    /// Specialized repeat when only a singleton final dimension expands.
    RepeatSingletonLast,
    /// Repeat-interleave on the contiguous final axis.
    RepeatInterleaveLast,
    RepeatInterleaveGeneric,
    ContiguousGather,
    ChannelsLastGather4d,
    ChannelsLastGather5d,
    FancyGatherOneAxis,
    MetadataAssignTensor,
    MetadataAssignScalar,
    Elementwise,
    ContiguousElementwise,
    BroadcastElementwise,
    /// Canonical U8 mask logical elementwise operations.
    MaskElementwise,
    /// Broadcast PReLU with a frozen channel/parameter mapping.
    ChannelwisePRelu,
    ScalarElementwise,
    /// Dedicated legacy-compatible p=1.5 scalar-power artifact.
    ScalarPowerThreeHalves,
    /// Ternary mask select/where artifact.
    Select,
    RowwiseReduction,
    TreeReduction,
    TwoPassReduction,
    RowwiseNormalization,
    FusedNormalization,
    FusedSoftmax,
    FusedLogSoftmax,
    ParallelPrefixScan,
    SegmentedScan,
    BitonicSort,
    ComparisonSort,
    /// Direct bounded-k selection without fully sorting the source axis.
    TopKSelection,
    /// Full-sort plus trim fallback for large k or target constraints.
    TopKSortTrim,
    /// Fully unrolled GPU candidate for contiguous final-axis FFT widths 2/4.
    FftSmallLastDimGpu,
    /// Radix-2 Cooley-Tukey candidate for power-of-two 1-D FFT extents.
    FftRadix2,
    /// Bluestein/chirp-z candidate for exact arbitrary-length 1-D FFT extents.
    FftBluestein,
    /// Unrolled/static solve-inverse-Cholesky candidate for 2x2/3x3 batches.
    CpuLinalgSmallStatic,
    /// Generic batched dense linear-algebra candidate with static sharding.
    CpuLinalgBatched,
    /// SIMD dot-product candidate for canonical 1-D x 1-D matmul.
    CpuDotSimd,
    /// SIMD dot-per-column vector x matrix candidate for generic strided matrices.
    CpuVecMatSimd,
    /// SIMD AXPY-style vector x row-major matrix candidate.
    CpuVecMatAxpy,
    /// SIMD row-dot matrix x vector candidate.
    CpuMatVecSimd,
    /// Fully unrolled fixed-shape CPU GEMM candidate for 2x2x2 and 3x3x3.
    CpuSgemmFixedTiny,
    /// Equal-batch small CPU GEMM (all M/K/N <= 64), statically sharded by batch.
    CpuSgemmBatchedSmall,
    CpuSgemmDirect,
    CpuSgemmDirectAxpy,
    CpuSgemmPacked4x8,
    /// Packed CPU SGEMM candidate using six output rows by one native SIMD-width
    /// column vector. Kept as an offline competitor to 4x8, never runtime-selected.
    CpuSgemmPacked6xSimd,
    /// Persistent fixed-weight path: B is packed once at binding/load time and
    /// replay packs only A before entering the frozen 4x8 microkernel.
    CpuSgemmPrepackedB4x8,
    /// Exact provider algorithm selected offline with provider heuristics disabled.
    VendorExact,
    /// Target-specific family whose semantics are carried by the frozen artifact.
    TargetNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelAssumptionPlan {
    /// If true, every physical IO storage range is pairwise non-overlapping.
    /// AOT kernels may then use noalias/restrict-like assumptions safely.
    pub noalias_io: bool,
    /// Minimum byte alignment assumed by vector/native loads; 0 means no stronger assumption.
    pub aligned_access_bytes: u16,
    /// The selected artifact may remove bounds checks because offline verification proved them.
    pub unchecked_indexing: bool,
    /// The selected artifact may use `core::hint::assert_unchecked`/equivalent verified assumptions.
    pub assert_unchecked_invariants: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkRange {
    pub begin: u64,
    pub end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuWorkerPlan {
    pub worker_id: u16,
    pub processor_group: u16,
    pub logical_cpu_id: u16,
    pub numa_node: u16,
    pub work: WorkRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuVectorAccessKind {
    Scalar,
    AlignedVector,
    UnalignedVector,
}

/// Operand-specific CPU memory policy selected offline. A single kernel can
/// therefore prefetch weights differently from activations and can stream large
/// outputs without forcing the same policy on every storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuMemoryAccessPlan {
    pub storage: StorageId,
    /// Zero for scalar access; otherwise the exact vector transaction width.
    pub vector_bytes: u16,
    pub vector_kind: CpuVectorAccessKind,
    /// Minimum address alignment assumed by the selected artifact.
    pub required_alignment_bytes: u16,
    pub prefetch_kind: PrefetchKind,
    pub prefetch_distance_bytes: u32,
    pub store_policy: StorePolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuKernelPlan {
    pub io: TargetIoPlan,
    pub workspace_slots: &'static [MemorySlotId],
    pub family: KernelFamily,
    pub artifact: ArtifactRef,
    pub rust_codegen: Option<RustCodegenPlan>,
    pub library_algorithm: Option<LibraryAlgorithmPlan>,
    pub required_features: FeatureMask,
    pub assumptions: KernelAssumptionPlan,
    pub math_mode: KernelMathMode,
    pub parallel_kind: CpuParallelKind,
    pub dispatch_mode: CpuDispatchMode,
    pub workers: &'static [CpuWorkerPlan],
    /// Exact per-storage CPU load/store/prefetch strategy. Empty means the
    /// kernel-wide defaults below are used for all operands.
    pub memory_access: &'static [CpuMemoryAccessPlan],
    pub simd_bytes: u16,
    pub vector_unroll: u8,
    pub scalar_unroll: u8,
    pub software_pipeline_stages: u8,
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub micro_m: u16,
    pub micro_n: u16,
    pub micro_k: u16,
    pub prefetch_kind: PrefetchKind,
    pub prefetch_distance_bytes: u32,
    pub pack_a: bool,
    pub pack_b: bool,
    pub persistent_packed_weights: bool,
    pub pack_alignment_bytes: u32,
    pub numa_policy: NumaPolicy,
    pub fixed_numa_node: u16,
    pub store_policy: StorePolicy,
    pub worker_wait: WorkerWaitPolicy,
}

// =============================================================================
// GPU mapping: all performance decisions are frozen offline
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuKernelSource {
    RustCudaAot,
    CudaCppAot,
    PtxAssemblyAot,
    ExternalNativeBinary,
    VendorLibrary,
}

/// Immutable provenance/codegen contract for a target-native GPU artifact.
/// `target_arch` carries architecture qualifiers such as sm_90a/sm_100a when
/// architecture-specific instructions require them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuCodegenPlan {
    pub source: GpuKernelSource,
    pub toolchain_version: &'static str,
    pub target_arch: &'static str,
    pub target_compute_major: u8,
    pub target_compute_minor: u8,
    pub ptx_isa_major: u8,
    pub ptx_isa_minor: u8,
    /// Actual resource usage of the selected compiled artifact, captured offline.
    pub compiled_registers_per_thread: u16,
    pub compiled_static_shared_bytes: u32,
    pub compiled_local_bytes_per_thread: u32,
    pub compiled_max_threads_per_block: u16,
    /// Exact active-block occupancy measured/queried offline for this artifact and launch shape.
    pub compiled_active_blocks_per_sm: u8,
    /// Optional compiler register ceiling used while producing the artifact.
    pub max_registers: u16,
    /// Frozen CUDA/NVVM math/codegen choices; never selected during replay.
    pub use_fast_math: bool,
    pub flush_denormals_to_zero: bool,
    pub precise_division: bool,
    pub precise_sqrt: bool,
    pub fuse_multiply_add: bool,
    pub extra_device_vectorization: bool,
    /// Device link-time optimization used when producing the final native artifact.
    pub device_lto: bool,
    pub relocatable_device_code: bool,
    /// Frozen ptxas optimization level. Maximum-runtime-performance artifacts use level 3.
    pub ptxas_opt_level: u8,
    /// Permit expensive optimizer passes; normally enabled by ptxas at O2/O3.
    pub allow_expensive_optimizations: bool,
    /// True only when nvcc/ptxas fast-compile is disabled (`-Ofc=0`).
    pub fast_compile_disabled: bool,
    /// Offline CompileIQ / nvcc advanced-controls tuning was applied to this exact artifact.
    pub compileiq_tuned: bool,
    /// Digest of the exact advanced-controls payload. Canonical zero when CompileIQ is unused.
    pub advanced_controls_hash_hi: u64,
    pub advanced_controls_hash_lo: u64,
    /// Device debug instrumentation must be absent from maximum-throughput artifacts.
    pub device_debug: bool,
    /// Stack canaries add device instructions and are excluded from the speed winner.
    pub device_stack_protector: bool,
    /// tcgen05 tensor-memory debug access checks are excluded from release winners.
    pub tensor_memory_access_checks: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Exact geometric cardinality. Untrusted/catalog-provided dimensions
    /// must be checked before they participate in any resource or launch proof.
    #[inline(always)]
    pub const fn checked_product(self) -> Option<u64> {
        let Some(xy) = (self.x as u64).checked_mul(self.y as u64) else {
            return None;
        };
        xy.checked_mul(self.z as u64)
    }

    /// Saturating display/diagnostic count. Never use this method to certify
    /// a launch: checked_product is mandatory in every validator.
    #[inline(always)]
    pub const fn product(self) -> u64 {
        match self.checked_product() {
            Some(count) => count,
            None => u64::MAX,
        }
    }

    #[inline(always)]
    pub const fn is_all_zero(self) -> bool {
        self.x == 0 && self.y == 0 && self.z == 0
    }

    #[inline(always)]
    pub const fn is_all_nonzero(self) -> bool {
        self.x != 0 && self.y != 0 && self.z != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AsyncCopyKind {
    None,
    SoftwarePrefetch,
    CpAsync,
    /// Hopper+ one-dimensional bulk-asynchronous global/shared copy. This is
    /// distinct from tensor-map TMA and freezes the cp.async.bulk path.
    CpAsyncBulk,
    Tma,
    TmaMulticast,
    Stas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapInterleave {
    None,
    Bytes16,
    Bytes32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapSwizzle {
    None,
    Bytes32,
    Bytes64,
    Bytes128,
    Bytes128Atom32,
    Bytes128Atom32Flip8,
    Bytes128Atom64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapL2Promotion {
    None,
    Bytes64,
    Bytes128,
    Bytes256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapOobFill {
    None,
    NanRequestZeroFma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapDirection {
    GlobalToShared,
    SharedToGlobal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapStoreReduction {
    None,
    Add,
    Min,
    Max,
    BitAnd,
    BitOr,
}

/// Fully frozen CUDA tensor-map descriptor for TMA (tiled or im2col). The
/// runtime binds the already-selected descriptor; it never infers mapping,
/// swizzles, traversal, or cache policy.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorMapKind {
    /// cuTensorMapEncodeTiled.
    Tiled,
    /// cuTensorMapEncodeIm2col (CC 9.0+).
    Im2Col,
    /// cuTensorMapEncodeIm2colWide in W mode (CC 10.0+).
    Im2ColWideW,
    /// cuTensorMapEncodeIm2colWide in W128 mode (CC 10.0+).
    Im2ColWideW128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorMapPlan {
    pub storage: StorageId,
    pub kind: TensorMapKind,
    pub global_dims: &'static [u64],
    /// CUDA tensor maps carry rank-1 global strides in bytes.
    pub global_strides_bytes: &'static [u64],
    /// Tiled traversal box. Must be empty for im2col tensor maps.
    pub box_dims: &'static [u32],
    pub element_strides: &'static [u32],
    /// Im2col spatial lower/upper offsets. For rank 3/4/5 the lengths are
    /// exactly 1/2/3. Empty for tiled maps and im2col-wide maps.
    pub im2col_lower: &'static [i32],
    pub im2col_upper: &'static [i32],
    /// Im2col-wide uses only W offsets; canonical zero otherwise.
    pub im2col_wide_lower: i32,
    pub im2col_wide_upper: i32,
    /// Im2col descriptor parameters; canonical zero for tiled maps.
    pub channels_per_pixel: u32,
    pub pixels_per_column: u32,
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapOobFill,
    pub direction: TensorMapDirection,
    /// Optional TMA reduction applied by shared-to-global stores. The exact
    /// supported dtype/operator pair is capability-gated and numerically verified offline.
    pub store_reduction: TensorMapStoreReduction,
    /// Hash of the offline-verified encoded tensor-map descriptor/configuration.
    pub descriptor_hash_hi: u64,
    pub descriptor_hash_lo: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BarrierKind {
    None,
    WarpSync,
    BlockBarrier,
    AsyncMBarrier,
    ClusterBarrier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MatrixAcceleration {
    None,
    TensorCore,
    Wmma,
    Mma,
    Wgmma,
    Tcgen05,
    BlockScaledMma,
    /// Structured-sparse mma.sp-style acceleration.
    SparseMma,
    /// Fifth-generation structured-sparse tcgen05.mma.sp path.
    SparseTcgen05,
    /// Structured-sparse block-scaled fifth-generation matrix acceleration.
    SparseBlockScaledMma,
    TargetNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuPartitionKind {
    Grid,
    SplitK,
    StreamK,
    PersistentCta,
    Grouped,
    Ragged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ClusterLaunchControlMode {
    Disabled,
    WorkStealing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GraphReplayMode {
    None,
    HostGraph,
    DeviceTailLaunch,
    DeviceFireAndForget,
    DeviceSiblingLaunch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SmPartitionKind {
    AllSms,
    GreenContext,
    ExecutionAffinity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ClusterSchedulingPolicy {
    NotApplicable,
    Spread,
    LoadBalancing,
    RubinDsmemLocality,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PortableClusterMode {
    NotApplicable,
    RequirePortable,
    AllowNonPortable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SharedMemoryMode {
    Standard,
    RequirePortable,
    AllowNonPortable,
    AllowOversized,
    PreferOversized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CudaSynchronizationPolicy {
    Spin,
    Yield,
    BlockingSync,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemorySyncDomainPlan {
    /// CUDA logical domain: 0 = default, 1 = remote.
    pub logical_domain: u8,
    /// Frozen logical-to-physical mapping selected offline.
    pub physical_domain: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum L2AccessProperty {
    Normal,
    Streaming,
    Persisting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct L2AccessPolicyPlan {
    pub storage: StorageId,
    pub offset_bytes: usize,
    pub bytes: usize,
    /// 0..=1000, representing CUDA's hit-ratio hint in thousandths.
    pub hit_ratio_per_mille: u16,
    pub hit: L2AccessProperty,
    pub miss: L2AccessProperty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuLoadCachePolicy {
    Default,
    CacheAll,
    CacheGlobal,
    Streaming,
    Volatile,
    NonCoherent,
    UniformReadOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuStoreCachePolicy {
    Default,
    WriteBack,
    CacheGlobal,
    Streaming,
    WriteThrough,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuCacheEvictionPriority {
    Default,
    Normal,
    EvictFirst,
    EvictLast,
    EvictUnchanged,
    NoAllocate,
}

/// Per-storage memory-instruction policy frozen into the selected GPU artifact.
/// This lets offline calibration choose cache operators/eviction hints and
/// vector widths for each operand without any replay-time policy branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuConstantMemoryPlan {
    /// Read-only storage copied/bound once into device constant memory.
    pub storage: StorageId,
    /// Byte offset in the frozen constant symbol/region used by the artifact.
    pub symbol_offset_bytes: u32,
    pub bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuMemoryAccessPlan {
    pub storage: StorageId,
    pub vector_bytes: u16,
    pub required_alignment_bytes: u16,
    /// Frozen PTX L2 prefetch-size hint for global loads/cp.async: 0, 64, 128 or 256 bytes.
    pub l2_prefetch_bytes: u16,
    pub load_cache: GpuLoadCachePolicy,
    pub store_cache: GpuStoreCachePolicy,
    pub l1_eviction: GpuCacheEvictionPriority,
    pub l2_eviction: GpuCacheEvictionPriority,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SmPartitionPlan {
    pub kind: SmPartitionKind,
    pub sm_count: u16,
    pub coscheduled_sm_count: u16,
    pub preferred_coscheduled_sm_count: u16,
    pub locality_domain: Option<u16>,
    pub allow_backfill: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuKernelPlan {
    pub io: TargetIoPlan,
    pub workspace_slots: &'static [MemorySlotId],
    pub family: KernelFamily,
    pub device_ordinal: u16,
    pub execution_context: ExecutionContextId,
    pub stream: StreamId,
    pub artifact: ArtifactRef,
    pub gpu_codegen: Option<GpuCodegenPlan>,
    pub library_algorithm: Option<LibraryAlgorithmPlan>,
    pub required_features: FeatureMask,
    pub assumptions: KernelAssumptionPlan,
    pub math_mode: KernelMathMode,
    pub grid: Dim3,
    pub block: Dim3,
    pub cluster: Dim3,
    pub preferred_cluster: Dim3,
    pub cluster_scheduling: ClusterSchedulingPolicy,
    pub portable_cluster_mode: PortableClusterMode,
    pub max_threads_per_block: u16,
    pub min_blocks_per_sm: u8,
    pub dynamic_shared_bytes: u32,
    pub shared_memory_mode: SharedMemoryMode,
    /// Blackwell+ Tensor Memory allocation, expressed in hardware TMEM columns.
    pub tmem_columns_per_cta: u16,
    pub tmem_accumulator_stages: u8,
    pub tmem_operand_a: bool,
    pub tmem_exclusive: bool,
    pub target_registers_per_thread: u16,
    pub target_blocks_per_sm: u8,
    pub persistent_ctas_per_sm: u8,
    pub vector_bytes: u16,
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub warp_m: u16,
    pub warp_n: u16,
    pub warp_k: u16,
    pub shared_swizzle: u8,
    pub shared_memory_carveout_percent: u8,
    pub pipeline_stages: u8,
    pub warps_per_block: u8,
    pub producer_warps: u8,
    pub consumer_warps: u8,
    pub matrix_acceleration: MatrixAcceleration,
    pub async_copy: AsyncCopyKind,
    /// Exact tensor-map descriptors when TMA/TMA multicast is selected. Multiple
    /// operands (for example GEMM A/B/C) may each carry a separately frozen map.
    pub tensor_maps: &'static [TensorMapPlan],
    pub barrier: BarrierKind,
    pub partition: GpuPartitionKind,
    pub split_k: u16,
    pub stream_k_slices: u16,
    pub grid_stride_loop: bool,
    pub cluster_launch_control: ClusterLaunchControlMode,
    pub use_distributed_shared_memory: bool,
    pub sm_partition: SmPartitionPlan,
    pub memory_sync_domain: Option<MemorySyncDomainPlan>,
    pub graph_replay: GraphReplayMode,
    /// Pre-upload the executable graph during one-time binding so the first
    /// deterministic replay does not pay graph-upload setup work.
    pub graph_preupload_stream: Option<StreamId>,
    /// Instantiate CUDA Graphs with node-priority semantics enabled. Required
    /// when a frozen per-kernel priority must remain effective inside a graph.
    pub graph_use_node_priority: bool,
    /// Graph-only CUDA launch attribute allowing device-side kernel-node parameter updates.
    pub device_updatable_graph_node: bool,
    /// Frozen single-instantiation contract required by CUDA device-updatable graph nodes.
    pub graph_single_instantiation: bool,
    /// Exact per-storage load/store/cache policy selected offline. Empty means
    /// the artifact uses its target-native defaults.
    pub memory_access: &'static [GpuMemoryAccessPlan],
    /// Small read-only operands promoted to the device constant cache. The
    /// copy/symbol binding is performed once; replay only launches the frozen artifact.
    pub constant_memory: &'static [GpuConstantMemoryPlan],
    pub l2_access_policy: Option<L2AccessPolicyPlan>,
    pub cooperative_launch: bool,
    pub cluster_launch: bool,
    pub programmatic_dependent_launch: bool,
    pub programmatic_stream_serialization: bool,
    pub programmatic_event: Option<EventId>,
    pub programmatic_event_trigger_at_block_start: bool,
    pub launch_completion_event: Option<EventId>,
    pub launch_priority: Option<i32>,
    pub nvlink_util_centric_scheduling: bool,
}

// =============================================================================
// Unified CPU / GPU / CPU+GPU node mapping
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Placement {
    Cpu,
    Gpu,
    CpuGpu,
}

/// Mapping mode for one canonical node that has both CPU and GPU placements.
/// This is placement metadata, not a separate runtime or subarchitecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuGpuMode {
    Partitioned,
    Pipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
}

/// One statically bounded slice of a CPU-GPU placement. `stage` is meaningful
/// for pipeline mode and is frozen offline together with schedule order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuGpuSlice {
    pub target: ExecutionTarget,
    pub work: WorkRange,
    pub stage: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuGpuPlacementPlan {
    pub mode: CpuGpuMode,
    pub partition_axis: u8,
    pub slices: &'static [CpuGpuSlice],
    /// Explicitly states whether CPU and GPU launches may overlap. The schedule
    /// still fixes launch order and completion points exactly.
    pub overlap_cpu_gpu: bool,
    pub pipeline_depth: u16,
}

/// Full semantic description of an operation inside a fused node. Every step
/// names its logical inputs and outputs so the frozen fusion can be checked
/// against the canonical static dataflow instead of being opaque metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedDataflowStep {
    /// Exact canonical semantic node represented by this fused step.
    pub semantic_node: NodeId,
    pub operation: Operation,
    pub inputs: &'static [TensorId],
    pub outputs: &'static [TensorId],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenNode {
    /// Execution-node id used by the frozen schedule.
    pub id: NodeId,
    /// Canonical semantic nodes covered by this execution node, in canonical
    /// topological order. A non-fused node contains exactly one id.
    pub semantic_nodes: &'static [NodeId],
    /// The first semantic operation. When `fused_operations` is non-empty it
    /// must equal the first fused step's operation.
    pub operation: Operation,
    pub dependencies: &'static [NodeId],
    pub inputs: &'static [TensorId],
    pub outputs: &'static [TensorId],
    pub fused_operations: &'static [FusedDataflowStep],
    pub placement: Placement,
    pub cpu: Option<&'static CpuKernelPlan>,
    pub gpu: Option<&'static GpuKernelPlan>,
    pub cpu_gpu: Option<CpuGpuPlacementPlan>,
}

/// Exact replay schedule. Combined CPU-GPU nodes are decomposed into explicit
/// slice launches plus an explicit completion point; transfers and sync records
/// may be interleaved between those launches. A runtime therefore never invents
/// partition ordering or a CPU/GPU overlap strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleItem {
    /// Complete execution of a CPU-only or GPU-only node.
    Node(NodeId),
    /// Dispatch one exact slice of a combined CPU-GPU placement.
    CpuGpuSliceLaunch {
        node: NodeId,
        slice: u16,
    },
    /// Explicit completion/wait point for that exact slice. The launch/complete
    /// pairs make overlap and sequencing replayable without backend inference.
    CpuGpuSliceComplete {
        node: NodeId,
        slice: u16,
    },
    Transfer(TransferId),
    TransferBatch(TransferBatchId),
    Sync(SyncId),
}

/// One contiguous GPU-only schedule interval materialized as a single CUDA
/// Graph executable during one-time binding. The range is [schedule_begin,
/// schedule_end) in the immutable schedule. Metadata-only AliasView items in
/// that range contribute dependencies/storage mappings but no CUDA graph node.
/// Replay launches exactly this graph and performs no capture, graph update,
/// policy choice, or per-node dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuGraphSegmentPlan {
    pub id: u32,
    pub schedule_begin: u32,
    pub schedule_end: u32,
    pub device_ordinal: u16,
    pub execution_context: ExecutionContextId,
    pub launch_stream: StreamId,
    pub mode: GraphReplayMode,
    pub preupload_stream: Option<StreamId>,
    pub use_node_priority: bool,
    pub single_instantiation: bool,
}

// =============================================================================
// Static resources, storage, transfers, and synchronization
// =============================================================================

// =============================================================================
// Static memory plan: arenas, slots and tensor residencies
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryClass {
    HostAligned,
    HostPinned,
    HostMappedZeroCopy,
    /// Ordinary pageable host allocation directly addressable by a coherent GPU
    /// using host page tables. Candidate generation must benchmark this against
    /// pinned/mapped/unified/device-resident alternatives; it is never assumed faster.
    HostPageableDirect,
    DeviceLocal,
    UnifiedManaged,
    External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HostMemoryPolicy {
    Pageable,
    Pinned,
    PinnedPortable,
    PinnedWriteCombined,
    MappedZeroCopy,
    MappedWriteCombined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeviceAllocationPolicy {
    NotApplicable,
    Preallocated,
    StreamOrderedPool,
    GraphOwned,
    VirtualMemoryMapped,
    /// VMM multicast mapping backed by a fixed multi-GPU multicast team.
    MulticastMapped,
    FabricMapped,
    ExtendedGpuMemory,
    External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryPoolPlan {
    pub release_threshold_bytes: u64,
    pub reuse_follow_event_dependencies: bool,
    pub reuse_allow_opportunistic: bool,
    pub reuse_allow_internal_dependencies: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LifetimeClass {
    Input,
    Output,
    Temporary,
    Persistent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResidencyPolicy {
    ReplayLocal,
    Persistent,
    PersistentWeight,
    ExternalBound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StorageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    Host,
    HostNuma(u16),
    Device(u16),
}

/// Frozen `cudaMemAdvise`-style guidance for unified/managed allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnifiedMemoryAdvicePlan {
    pub read_mostly: bool,
    pub preferred_location: Option<MemoryLocation>,
    pub accessed_by: &'static [MemoryLocation],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryArenaPlan {
    pub id: ArenaId,
    pub class: MemoryClass,
    pub bytes: usize,
    pub alignment_bytes: usize,
    /// Fixed NUMA node for a host arena; mutually exclusive with numa_interleave_nodes.
    pub numa_node: Option<u16>,
    /// Exact NUMA interleave set selected offline for memory-bound CPU workloads.
    pub numa_interleave_nodes: &'static [u16],
    /// Exact/verified host page size for huge-page policies; zero with PagePolicy::BasePages.
    pub page_size_bytes: usize,
    /// Touch/map all host pages during one-time binding so replay does not fault them in.
    pub prefault: bool,
    pub device_ordinal: Option<u16>,
    pub locality_domain: Option<u16>,
    /// Empty for ordinary arenas. For MulticastMapped arenas this is the exact
    /// frozen CUDA multicast team and must include device_ordinal.
    pub multicast_devices: &'static [u16],
    pub host_policy: HostMemoryPolicy,
    pub device_allocation: DeviceAllocationPolicy,
    pub memory_pool: Option<MemoryPoolPlan>,
    pub page_policy: PagePolicy,
    pub unified_advice: Option<UnifiedMemoryAdvicePlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemorySlotPlan {
    pub id: MemorySlotId,
    pub arena: ArenaId,
    pub offset_bytes: usize,
    pub bytes: usize,
    pub alignment_bytes: usize,
    pub lifetime: LifetimeClass,
    pub live_from_step: u32,
    pub live_until_step: u32,
    pub reuse_group: u32,
    pub alias_group: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorStoragePlan {
    pub id: StorageId,
    pub tensor: TensorId,
    pub slot: MemorySlotId,
    /// Byte offset from the memory-slot start to logical element [0, ...].
    /// Together with signed/zero strides this represents narrow, reverse,
    /// transpose, broadcast, arbitrary as-strided and sliding-window aliases
    /// without materialization.
    pub base_offset_bytes: usize,
    /// For an alias view over an externally supplied buffer, identifies the
    /// one root storage that owns the runtime binding. Derived read-only views
    /// need no duplicate pointer binding. `None` is canonical otherwise.
    pub external_alias_of: Option<StorageId>,
    /// Exact physical layout for this residency. CPU-packed and GPU-native
    /// copies of one logical tensor may therefore coexist without ambiguity.
    pub physical_layout: TensorLayout,
    /// Exact number of bytes occupied by this physical representation.
    pub physical_bytes: usize,
    /// Equal tags mean raw byte-wise transfer is representation-compatible.
    pub format_tag: u64,
    pub access: StorageAccess,
    pub residency: ResidencyPolicy,
}

/// Target-specific physical bindings for one logical node. The index in each
/// storage array corresponds exactly to the same index in FrozenNode inputs or
/// outputs. This keeps the canonical graph logical while allowing CPU and GPU
/// to use different packed/resident representations of the same tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetIoPlan {
    pub input_storages: &'static [StorageId],
    pub output_storages: &'static [StorageId],
}

// =============================================================================
// Streams, execution contexts, transfer and synchronization
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WorkqueueSharingScope {
    DeviceContext,
    GreenContextBalanced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkqueuePlan {
    /// Expected maximum number of concurrent stream-ordered workloads.
    pub concurrency_limit: u16,
    pub sharing_scope: WorkqueueSharingScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuExecutionContextPlan {
    pub id: ExecutionContextId,
    pub device_ordinal: u16,
    pub sm_partition: SmPartitionPlan,
    /// Exact L2 set-aside for persisting accesses, configured once during binding.
    pub persisting_l2_reservation_bytes: usize,
    /// Optional CUDA green-context workqueue resource configuration.
    pub workqueue: Option<WorkqueuePlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamPlan {
    pub id: StreamId,
    pub device_ordinal: u16,
    pub execution_context: ExecutionContextId,
    pub priority: i32,
    pub non_blocking: bool,
    pub synchronization_policy: CudaSynchronizationPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EventPlan {
    pub id: EventId,
    pub device_ordinal: u16,
    pub disable_timing: bool,
    pub interprocess: bool,
}

/// Host-visible boundary classification frozen into a transfer. Internal
/// CPU<->GPU staging remains `None`; only data intentionally returned to a host
/// API receives a non-None boundary. This prevents accidental hot-path readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HostBoundaryClass {
    None,
    /// One exact integer/mask metadata scalar intentionally read back to the host.
    ScalarMetadataReadback,
    /// Explicit public host-output/readback API. The runtime primitive is completion-guaranteed.
    ExplicitHostApi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemcpySrcAccessOrder {
    /// Source accesses obey stream order.
    Stream,
    /// CUDA may consume the host source during the API call. Intended for
    /// short-lived/ephemeral host sources whose prior stream dependencies are absent.
    DuringApiCall,
    /// CUDA may access the host source independently of stream order, including
    /// after the API call returns. The source lifetime is therefore pinned by the
    /// bound execution object for the entire replay lifetime.
    Any,
}

/// Exact CUDA memcpy attributes selected offline. These are part of the frozen
/// transfer identity so `cudaMemcpyBatchAsync`/attribute-aware async copies never
/// infer source ordering, locality hints, or overlap policy during replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemcpyAttributePlan {
    pub src_access_order: MemcpySrcAccessOrder,
    pub src_location_hint: Option<MemoryLocation>,
    pub dst_location_hint: Option<MemoryLocation>,
    pub prefer_overlap_with_compute: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TransferKind {
    None,
    DeviceResident,
    AsyncPinnedCopy,
    /// Direct pointer copy involving ordinary system-pageable host memory.
    /// This remains a benchmarked alternative to registration/pinning/staging;
    /// depending on size, reuse and platform it may win by avoiding setup/copies.
    PageableHostCopy,
    MappedPinnedZeroCopy,
    PageableDirectZeroCopy,
    UnifiedPrefetch,
    /// cudaMemPrefetchAsync over ordinary system-allocated memory on devices
    /// that support pageable-memory prefetch. The storage remains a single
    /// allocation; this is migration/locality guidance, not a copy.
    PageableSystemPrefetch,
    /// Discard stale contents of an in-place managed/system-pageable range before
    /// a later overwrite. No bytes are copied and reading before a subsequent
    /// write/prefetch is invalid by contract.
    ManagedDiscard,
    /// Combined discard+prefetch for an in-place managed/system-pageable range.
    /// CUDA exposes this as a single operation that is more efficient than
    /// issuing discard and prefetch separately.
    ManagedDiscardAndPrefetch,
    DeviceToDevice,
    PeerToPeer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferPlan {
    pub id: TransferId,
    pub host_boundary: HostBoundaryClass,
    pub tensor: TensorId,
    pub src_storage: StorageId,
    pub dst_storage: StorageId,
    pub kind: TransferKind,
    /// Frozen attributes for CUDA pointer-copy primitives. Required for members
    /// of an `AsyncCopy` batch because CUDA requires a valid source-access-order
    /// attribute for every batch member. Must be absent for non-copy transfers.
    pub copy_attributes: Option<MemcpyAttributePlan>,
    pub stream: StreamId,
    pub src_offset_bytes: usize,
    pub dst_offset_bytes: usize,
    pub bytes: usize,
    pub chunk_bytes: usize,
    /// Reserved. Transfer overlap/concurrency is represented only by the exact
    /// schedule and `TransferBatchPlan`; this field must be zero.
    pub overlap_group: u32,
    pub producer_node: Option<NodeId>,
    /// Optional exact combined-placement slice. `None` means the complete node.
    pub producer_slice: Option<u16>,
    pub consumer_node: Option<NodeId>,
    /// Optional exact combined-placement slice. `None` means the complete node.
    pub consumer_slice: Option<u16>,
    pub prefetch_location: Option<MemoryLocation>,
}

/// Exact native batch primitive selected offline. Batch members have no ordering
/// dependency on one another; the batch itself occupies one schedule step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TransferBatchKind {
    /// CUDA-13-style cudaMemcpyBatchAsync.
    AsyncCopy,
    /// CUDA-13.4-style cudaMemPrefetchBatchAsync for managed/pageable-system ranges.
    MemoryPrefetch,
    /// CUDA driver batched discard for managed/system-pageable ranges.
    MemoryDiscard,
    /// CUDA driver batched discard+prefetch primitive.
    MemoryDiscardAndPrefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferBatchPlan {
    pub id: TransferBatchId,
    pub kind: TransferBatchKind,
    pub stream: StreamId,
    pub transfers: &'static [TransferId],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SyncKind {
    None,
    CpuBarrier,
    GpuEventRecord,
    StreamWaitEvent,
    CpuWaitGpuEvent,
    GpuWaitExternal,
    ProgrammaticDependency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncPlan {
    pub id: SyncId,
    pub kind: SyncKind,
    pub event: Option<EventId>,
    pub barrier: Option<BarrierId>,
    pub producer_node: NodeId,
    pub producer_slice: Option<u16>,
    pub consumer_node: NodeId,
    pub consumer_slice: Option<u16>,
    pub stream: Option<StreamId>,
}

// =============================================================================
// Immutable offline-selection evidence
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CandidateId {
    pub hi: u64,
    pub lo: u64,
}

impl CandidateId {
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        (self.hi | self.lo) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CostEstimate {
    pub score: u64,
    pub workspace_bytes: usize,
    pub estimated_register_bytes: u32,
    pub estimated_shared_memory_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BenchmarkStats {
    pub warmup_samples: u32,
    pub measured_samples: u32,
    pub median_ns: u64,
    pub mad_ns: u64,
    pub p95_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub rejected_outliers: u32,
}

impl BenchmarkStats {
    #[inline]
    pub fn is_valid(self) -> bool {
        // A measured host-observed E2E latency rounded to zero nanoseconds
        // is below the instrument's usable resolution, not verified evidence
        // of a zero-cost native operation. Fail closed before winner selection.
        self.warmup_samples > 0
            && self.measured_samples >= 3
            && self.min_ns > 0
            && self.min_ns <= self.median_ns
            && self.median_ns <= self.p95_ns
            && self.p95_ns <= self.max_ns
            && self.mad_ns <= self.max_ns.saturating_sub(self.min_ns)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NoiseCalibration {
    pub uncertainty_ns: u64,
    pub timer_resolution_ns: u64,
    pub environment_signature_hi: u64,
    pub environment_signature_lo: u64,
}

impl NoiseCalibration {
    #[inline]
    pub fn is_valid(self) -> bool {
        self.timer_resolution_ns > 0
            && self.uncertainty_ns >= self.timer_resolution_ns
            && (self.environment_signature_hi | self.environment_signature_lo) != 0
    }

    #[inline]
    pub fn digest(self) -> (u64, u64) {
        stable_hash_pair(&self, 0x4341_4c49_4252_4154)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericalVerification {
    pub passed: bool,
    pub reference_signature_hi: u64,
    pub reference_signature_lo: u64,
    pub max_abs_error_f32_bits: u32,
    pub max_rel_error_f32_bits: u32,
    /// Exact digest of reference non-F32 outputs (Index/Mask) over the frozen
    /// verification corpus. Zero when the workload produces no such outputs.
    pub exact_output_reference_hi: u64,
    pub exact_output_reference_lo: u64,
    /// Exact digest produced by the candidate over the same corpus.
    pub exact_output_candidate_hi: u64,
    pub exact_output_candidate_lo: u64,
}

impl NumericalVerification {
    #[inline]
    pub fn is_valid(self) -> bool {
        let abs = f32::from_bits(self.max_abs_error_f32_bits);
        let rel = f32::from_bits(self.max_rel_error_f32_bits);
        (self.reference_signature_hi | self.reference_signature_lo) != 0
            && abs.is_finite()
            && rel.is_finite()
            && abs >= 0.0
            && rel >= 0.0
    }

    #[inline]
    pub fn satisfies(self, workload: &CanonicalWorkload) -> bool {
        if !self.passed || !self.is_valid() {
            return false;
        }
        let reference = workload.verification_reference_signature();
        if self.reference_signature_hi != reference.0 || self.reference_signature_lo != reference.1
        {
            return false;
        }
        let observed_abs = f32::from_bits(self.max_abs_error_f32_bits);
        let observed_rel = f32::from_bits(self.max_rel_error_f32_bits);
        let (allowed_abs, allowed_rel) = workload.verification_tolerance();
        if observed_abs > allowed_abs || observed_rel > allowed_rel {
            return false;
        }

        let mut requires_exact_metadata = false;
        let mut tensor_index = 0usize;
        while tensor_index < workload.graph.tensors.len() && !requires_exact_metadata {
            let tensor = &workload.graph.tensors[tensor_index];
            if tensor.dtype != crate::tensor_light::tensor::DType::F32 {
                let mut node_index = 0usize;
                while node_index < workload.graph.nodes.len() {
                    if workload.graph.nodes[node_index]
                        .outputs
                        .contains(&tensor.id)
                    {
                        requires_exact_metadata = true;
                        break;
                    }
                    node_index += 1;
                }
            }
            tensor_index += 1;
        }

        let reference_exact = (
            self.exact_output_reference_hi,
            self.exact_output_reference_lo,
        );
        let candidate_exact = (
            self.exact_output_candidate_hi,
            self.exact_output_candidate_lo,
        );
        if requires_exact_metadata {
            reference_exact != (0, 0)
                && candidate_exact != (0, 0)
                && reference_exact == candidate_exact
        } else {
            reference_exact == (0, 0) && candidate_exact == (0, 0)
        }
    }

    #[inline]
    pub fn digest(self) -> (u64, u64) {
        stable_hash_pair(&self, 0x5645_5249_4659_4e55)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CandidateEvidence {
    pub id: CandidateId,
    pub capability_accepted: bool,
    pub cost: CostEstimate,
    pub benchmark: BenchmarkStats,
    pub calibration: NoiseCalibration,
    pub verification: NumericalVerification,
}

impl CandidateEvidence {
    #[inline]
    pub fn is_winner_eligible(self) -> bool {
        self.id.is_valid()
            && self.capability_accepted
            && self.benchmark.is_valid()
            && self.calibration.is_valid()
            && self.verification.is_valid()
            && self.verification.passed
    }
}

/// An explicit OFFLINE *held-out* host-observed end-to-end confirmation.
/// A trusted benchmark provider supplies independent measurement provenance;
/// this struct is a recorded evidence contract, not a cryptographic proof
/// that provider measurements were independent or hardware timing was honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeldOutConfirmation {
    pub passed: bool,
    /// Provider attests this was not the screening or candidate-search series.
    pub independent_series: bool,
    pub winner: CandidateId,
    pub strongest_challenger: Option<CandidateId>,
    pub winner_e2e: BenchmarkStats,
    pub challenger_e2e: Option<BenchmarkStats>,
    pub baseline_e2e: BenchmarkStats,
    /// Uncertainty of the fresh held-out series, NOT screening uncertainty.
    pub held_out_uncertainty_ns: u64,
    pub held_out_timer_resolution_ns: u64,
    pub environment_signature_hi: u64,
    pub environment_signature_lo: u64,
    /// Digest of the held-out run-order/measurement corpus and metadata.
    pub series_hash_hi: u64,
    pub series_hash_lo: u64,
}

impl HeldOutConfirmation {
    pub fn is_valid_for(
        self,
        screening: NoiseCalibration,
        expected_winner: CandidateId,
        expected_challenger: Option<CandidateId>,
    ) -> bool {
        let slack = self.held_out_uncertainty_ns;
        self.passed
            && self.independent_series
            && self.held_out_timer_resolution_ns > 0
            && slack >= self.held_out_timer_resolution_ns
            && self.winner_e2e.median_ns <= self.baseline_e2e.median_ns.saturating_add(slack)
            && self.challenger_e2e.is_none_or(|stats| {
                self.winner_e2e.median_ns <= stats.median_ns.saturating_add(slack)
            })
            && self.winner == expected_winner
            && self.winner.is_valid()
            && self.strongest_challenger == expected_challenger
            && self.winner_e2e.is_valid()
            && self.baseline_e2e.is_valid()
            && self.challenger_e2e.is_some() == expected_challenger.is_some()
            && self.challenger_e2e.is_none_or(BenchmarkStats::is_valid)
            && expected_challenger.is_none_or(|id| id.is_valid() && id != self.winner)
            && self.environment_signature_hi == screening.environment_signature_hi
            && self.environment_signature_lo == screening.environment_signature_lo
            && (self.series_hash_hi | self.series_hash_lo) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectionEvidence {
    pub workload: WorkloadSignature,
    pub winner: CandidateId,
    pub winner_evidence: CandidateEvidence,
    pub final_key: PlanKey,
    pub generated_candidates: u32,
    pub capability_survivors: u32,
    pub cost_model_survivors: u32,
    pub benchmarked_candidates: u32,
    pub noise_calibrated_candidates: u32,
    pub numerically_verified_candidates: u32,
    pub verification_passed_candidates: u32,
    pub candidate_set_hash_hi: u64,
    pub candidate_set_hash_lo: u64,
    pub held_out: HeldOutConfirmation,
    pub evidence_hash_hi: u64,
    pub evidence_hash_lo: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SelectionEvidencePayload {
    workload: WorkloadSignature,
    winner: CandidateId,
    winner_evidence: CandidateEvidence,
    final_key: PlanKey,
    generated_candidates: u32,
    capability_survivors: u32,
    cost_model_survivors: u32,
    benchmarked_candidates: u32,
    noise_calibrated_candidates: u32,
    numerically_verified_candidates: u32,
    verification_passed_candidates: u32,
    candidate_set_hash_hi: u64,
    candidate_set_hash_lo: u64,
    held_out: HeldOutConfirmation,
}

impl SelectionEvidence {
    /// Internal placeholder used only to construct a non-replayable plan view
    /// for offline structural preflight. Full winner evidence is still required
    /// by `freeze_offline` and runtime binding.
    #[cfg(feature = "aot")]
    pub(crate) fn preflight_placeholder(
        workload: WorkloadSignature,
        candidate: CandidateId,
    ) -> Self {
        Self::new(
            workload,
            CandidateEvidence {
                id: candidate,
                capability_accepted: true,
                cost: CostEstimate {
                    score: 0,
                    workspace_bytes: 0,
                    estimated_register_bytes: 0,
                    estimated_shared_memory_bytes: 0,
                },
                benchmark: BenchmarkStats {
                    warmup_samples: 1,
                    measured_samples: 3,
                    median_ns: 1,
                    mad_ns: 0,
                    p95_ns: 1,
                    min_ns: 1,
                    max_ns: 1,
                    rejected_outliers: 0,
                },
                calibration: NoiseCalibration {
                    uncertainty_ns: 1,
                    timer_resolution_ns: 1,
                    environment_signature_hi: 1,
                    environment_signature_lo: 1,
                },
                verification: NumericalVerification {
                    passed: true,
                    reference_signature_hi: 1,
                    reference_signature_lo: 1,
                    max_abs_error_f32_bits: 0,
                    max_rel_error_f32_bits: 0,
                    exact_output_reference_hi: 0,
                    exact_output_reference_lo: 0,
                    exact_output_candidate_hi: 0,
                    exact_output_candidate_lo: 0,
                },
            },
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            (1, 1),
            HeldOutConfirmation {
                passed: false,
                independent_series: false,
                winner: candidate,
                strongest_challenger: None,
                winner_e2e: BenchmarkStats {
                    warmup_samples: 1,
                    measured_samples: 3,
                    median_ns: 1,
                    mad_ns: 0,
                    p95_ns: 1,
                    min_ns: 1,
                    max_ns: 1,
                    rejected_outliers: 0,
                },
                challenger_e2e: None,
                baseline_e2e: BenchmarkStats {
                    warmup_samples: 1,
                    measured_samples: 3,
                    median_ns: 1,
                    mad_ns: 0,
                    p95_ns: 1,
                    min_ns: 1,
                    max_ns: 1,
                    rejected_outliers: 0,
                },
                held_out_uncertainty_ns: 1,
                held_out_timer_resolution_ns: 1,
                environment_signature_hi: 1,
                environment_signature_lo: 1,
                series_hash_hi: 0,
                series_hash_lo: 0,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workload: WorkloadSignature,
        winner_evidence: CandidateEvidence,
        generated_candidates: u32,
        capability_survivors: u32,
        cost_model_survivors: u32,
        benchmarked_candidates: u32,
        noise_calibrated_candidates: u32,
        numerically_verified_candidates: u32,
        verification_passed_candidates: u32,
        candidate_set_hash: (u64, u64),
        held_out: HeldOutConfirmation,
    ) -> Self {
        let final_key = workload.plan_key();
        let payload = SelectionEvidencePayload {
            workload,
            winner: winner_evidence.id,
            winner_evidence,
            final_key,
            generated_candidates,
            capability_survivors,
            cost_model_survivors,
            benchmarked_candidates,
            noise_calibrated_candidates,
            numerically_verified_candidates,
            verification_passed_candidates,
            candidate_set_hash_hi: candidate_set_hash.0,
            candidate_set_hash_lo: candidate_set_hash.1,
            held_out,
        };
        let digest = stable_hash_pair(&payload, 0x5345_4c45_4354_4556);
        Self {
            workload: payload.workload,
            winner: payload.winner,
            winner_evidence: payload.winner_evidence,
            final_key: payload.final_key,
            generated_candidates: payload.generated_candidates,
            capability_survivors: payload.capability_survivors,
            cost_model_survivors: payload.cost_model_survivors,
            benchmarked_candidates: payload.benchmarked_candidates,
            noise_calibrated_candidates: payload.noise_calibrated_candidates,
            numerically_verified_candidates: payload.numerically_verified_candidates,
            verification_passed_candidates: payload.verification_passed_candidates,
            candidate_set_hash_hi: payload.candidate_set_hash_hi,
            candidate_set_hash_lo: payload.candidate_set_hash_lo,
            held_out: payload.held_out,
            evidence_hash_hi: digest.0,
            evidence_hash_lo: digest.1,
        }
    }

    #[inline]
    pub fn expected_evidence_hash(self) -> (u64, u64) {
        let payload = SelectionEvidencePayload {
            workload: self.workload,
            winner: self.winner,
            winner_evidence: self.winner_evidence,
            final_key: self.final_key,
            generated_candidates: self.generated_candidates,
            capability_survivors: self.capability_survivors,
            cost_model_survivors: self.cost_model_survivors,
            benchmarked_candidates: self.benchmarked_candidates,
            noise_calibrated_candidates: self.noise_calibrated_candidates,
            numerically_verified_candidates: self.numerically_verified_candidates,
            verification_passed_candidates: self.verification_passed_candidates,
            candidate_set_hash_hi: self.candidate_set_hash_hi,
            candidate_set_hash_lo: self.candidate_set_hash_lo,
            held_out: self.held_out,
        };
        stable_hash_pair(&payload, 0x5345_4c45_4354_4556)
    }

    pub fn is_self_consistent(self) -> bool {
        let expected = self.expected_evidence_hash();
        self.workload.is_complete()
            && self.winner.is_valid()
            && self.winner == self.winner_evidence.id
            && self.winner_evidence.is_winner_eligible()
            && self.final_key == self.workload.plan_key()
            && self.generated_candidates > 0
            && self.capability_survivors > 0
            && self.cost_model_survivors > 0
            && self.benchmarked_candidates > 0
            && self.noise_calibrated_candidates == self.benchmarked_candidates
            // Every cost survivor is numerically checked; only VERIFIED
            // candidates may be benchmarked, never rejected ones.
            && self.numerically_verified_candidates == self.cost_model_survivors
            && self.verification_passed_candidates > 0
            && self.verification_passed_candidates == self.benchmarked_candidates
            && self.verification_passed_candidates <= self.numerically_verified_candidates
            && self.benchmarked_candidates <= self.cost_model_survivors
            && self.held_out.is_valid_for(
                self.winner_evidence.calibration,
                self.winner,
                self.held_out.strongest_challenger,
            )
            && self.cost_model_survivors <= self.capability_survivors
            && self.capability_survivors <= self.generated_candidates
            && (self.candidate_set_hash_hi | self.candidate_set_hash_lo) != 0
            && self.evidence_hash_hi == expected.0
            && self.evidence_hash_lo == expected.1
    }

    #[inline]
    pub fn calibration_hash(self) -> (u64, u64) {
        self.winner_evidence.calibration.digest()
    }

    #[inline]
    pub fn verification_hash(self) -> (u64, u64) {
        self.winner_evidence.verification.digest()
    }
}

// =============================================================================
// Frozen plan identity and materialization
// =============================================================================

// =============================================================================
// Frozen execution plan and exact plan identity
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanSeal {
    pub plan_hash_hi: u64,
    pub plan_hash_lo: u64,
    pub calibration_hash_hi: u64,
    pub calibration_hash_lo: u64,
    pub verification_hash_hi: u64,
    pub verification_hash_lo: u64,
}

/// Candidate-specific execution payload. Canonical semantics, tensors and the
/// hardware contract are intentionally absent: they come only from the
/// CanonicalWorkload and therefore cannot be replaced by a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenPlanParts {
    pub name: &'static str,
    pub arenas: &'static [MemoryArenaPlan],
    pub memory_slots: &'static [MemorySlotPlan],
    pub storages: &'static [TensorStoragePlan],
    pub execution_contexts: &'static [GpuExecutionContextPlan],
    pub streams: &'static [StreamPlan],
    pub events: &'static [EventPlan],
    pub nodes: &'static [FrozenNode],
    pub transfers: &'static [TransferPlan],
    pub transfer_batches: &'static [TransferBatchPlan],
    pub synchronization: &'static [SyncPlan],
    pub schedule: &'static [ScheduleItem],
    pub gpu_graph_segments: &'static [GpuGraphSegmentPlan],
}

impl FrozenPlanParts {
    /// Content identity for the performance-relevant execution payload.
    /// `name` is deliberately excluded so a duplicate candidate cannot evade
    /// deduplication merely by changing a display label.
    pub fn candidate_id(&self) -> CandidateId {
        let mut hi = StableIdentityHasher::new(0x4341_4e44_504c_414e); // CANDPLAN
        let mut lo = StableIdentityHasher::new(0x504c_414e_4341_4e44); // PLANCAND

        macro_rules! hash_field {
            ($value:expr) => {{
                $value.hash(&mut hi);
                $value.hash(&mut lo);
            }};
        }

        hash_field!(self.arenas);
        hash_field!(self.memory_slots);
        hash_field!(self.storages);
        hash_field!(self.execution_contexts);
        hash_field!(self.streams);
        hash_field!(self.events);
        hash_field!(self.nodes);
        hash_field!(self.transfers);
        hash_field!(self.transfer_batches);
        hash_field!(self.synchronization);
        hash_field!(self.schedule);
        hash_field!(self.gpu_graph_segments);

        CandidateId {
            hi: hi.finish(),
            lo: lo.finish(),
        }
    }

    /// Central offline preflight of the exact materialized candidate payload.
    /// This runs before capability gating, costing or benchmarking, preventing
    /// malformed mappings from entering any performance/capability decision stage.
    #[cfg(feature = "aot")]
    pub fn validate_for(&self, canonical: CanonicalWorkload) -> Result<(), PlanError> {
        if self.name.is_empty() {
            return Err(PlanError::InvalidCandidateIdentity);
        }
        canonical.validate()?;
        let workload = canonical.signature();
        let candidate = self.candidate_id();
        if !candidate.is_valid() || !workload.is_complete() {
            return Err(PlanError::InvalidCandidateIdentity);
        }
        let plan = FrozenExecutionPlan {
            name: self.name,
            key: workload.plan_key(),
            seal: PlanSeal {
                plan_hash_hi: 1,
                plan_hash_lo: 1,
                calibration_hash_hi: 1,
                calibration_hash_lo: 1,
                verification_hash_hi: 1,
                verification_hash_lo: 1,
            },
            canonical,
            workload,
            evidence: SelectionEvidence::preflight_placeholder(workload, candidate),
            hardware: canonical.hardware,
            tensors: canonical.graph.tensors,
            arenas: self.arenas,
            memory_slots: self.memory_slots,
            storages: self.storages,
            execution_contexts: self.execution_contexts,
            streams: self.streams,
            events: self.events,
            nodes: self.nodes,
            transfers: self.transfers,
            transfer_batches: self.transfer_batches,
            synchronization: self.synchronization,
            schedule: self.schedule,
            gpu_graph_segments: self.gpu_graph_segments,
        };
        plan.validate_static_payload()
    }
}

/// Portable Rust representation of one already-selected frozen plan.
///
/// Offline tooling can emit this value as Rust source together with its static
/// arrays/artifact descriptors. A runtime-only build can then reconstruct the
/// exact `FrozenExecutionPlan` with `from_materialized_image` without linking
/// candidate generation, benchmarking, cost models, or any other offline code.
/// The image is still fully seal/evidence validated before execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenPlanImage {
    pub key: PlanKey,
    pub seal: PlanSeal,
    pub canonical: CanonicalWorkload,
    pub workload: WorkloadSignature,
    pub evidence: SelectionEvidence,
    pub parts: FrozenPlanParts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenExecutionPlan {
    pub(crate) name: &'static str,
    pub(crate) key: PlanKey,
    pub(crate) seal: PlanSeal,
    pub(crate) canonical: CanonicalWorkload,
    pub(crate) workload: WorkloadSignature,
    pub(crate) evidence: SelectionEvidence,
    pub(crate) hardware: HardwareContract,
    pub(crate) tensors: &'static [TensorSpec],
    pub(crate) arenas: &'static [MemoryArenaPlan],
    pub(crate) memory_slots: &'static [MemorySlotPlan],
    pub(crate) storages: &'static [TensorStoragePlan],
    pub(crate) execution_contexts: &'static [GpuExecutionContextPlan],
    pub(crate) streams: &'static [StreamPlan],
    pub(crate) events: &'static [EventPlan],
    pub(crate) nodes: &'static [FrozenNode],
    pub(crate) transfers: &'static [TransferPlan],
    pub(crate) transfer_batches: &'static [TransferBatchPlan],
    pub(crate) synchronization: &'static [SyncPlan],
    pub(crate) schedule: &'static [ScheduleItem],
    pub(crate) gpu_graph_segments: &'static [GpuGraphSegmentPlan],
}

impl FrozenExecutionPlan {
    /// Reconstruct an immutable plan image emitted by offline specialization.
    /// This performs no runtime selection. `PlanDb`/binding validates the seal,
    /// evidence, canonical workload, resources, and exact target contracts before replay.
    #[inline(always)]
    pub const fn from_materialized_image(image: FrozenPlanImage) -> Self {
        Self {
            name: image.parts.name,
            key: image.key,
            seal: image.seal,
            canonical: image.canonical,
            workload: image.workload,
            evidence: image.evidence,
            hardware: image.canonical.hardware,
            tensors: image.canonical.graph.tensors,
            arenas: image.parts.arenas,
            memory_slots: image.parts.memory_slots,
            storages: image.parts.storages,
            execution_contexts: image.parts.execution_contexts,
            streams: image.parts.streams,
            events: image.parts.events,
            nodes: image.parts.nodes,
            transfers: image.parts.transfers,
            transfer_batches: image.parts.transfer_batches,
            synchronization: image.parts.synchronization,
            schedule: image.parts.schedule,
            gpu_graph_segments: image.parts.gpu_graph_segments,
        }
    }

    /// Export the exact immutable representation that an offline code generator
    /// can emit into a runtime-only Rust build.
    #[inline(always)]
    pub const fn materialized_image(&self) -> FrozenPlanImage {
        FrozenPlanImage {
            key: self.key,
            seal: self.seal,
            canonical: self.canonical,
            workload: self.workload,
            evidence: self.evidence,
            parts: FrozenPlanParts {
                name: self.name,
                arenas: self.arenas,
                memory_slots: self.memory_slots,
                storages: self.storages,
                execution_contexts: self.execution_contexts,
                streams: self.streams,
                events: self.events,
                nodes: self.nodes,
                transfers: self.transfers,
                transfer_batches: self.transfer_batches,
                synchronization: self.synchronization,
                schedule: self.schedule,
                gpu_graph_segments: self.gpu_graph_segments,
            },
        }
    }

    /// Sole plan materialization path after offline winner selection.
    #[cfg(feature = "aot")]
    pub fn freeze_offline(
        parts: FrozenPlanParts,
        canonical: CanonicalWorkload,
        evidence: SelectionEvidence,
    ) -> Result<Self, PlanError> {
        canonical.validate()?;
        let workload = canonical.signature();
        let key = workload.plan_key();
        let candidate_id = parts.candidate_id();
        if !candidate_id.is_valid()
            || evidence.winner != candidate_id
            || !workload.is_complete()
            || evidence.workload != workload
            || evidence.final_key != key
            || !evidence.is_self_consistent()
            || !evidence.winner_evidence.verification.satisfies(&canonical)
        {
            return Err(PlanError::InvalidPlanEvidence);
        }

        let calibration = evidence.calibration_hash();
        let verification = evidence.verification_hash();
        if (calibration.0 | calibration.1) == 0 || (verification.0 | verification.1) == 0 {
            return Err(PlanError::InvalidPlanSeal);
        }

        let mut plan = Self {
            name: parts.name,
            key,
            seal: PlanSeal {
                plan_hash_hi: 0,
                plan_hash_lo: 0,
                calibration_hash_hi: calibration.0,
                calibration_hash_lo: calibration.1,
                verification_hash_hi: verification.0,
                verification_hash_lo: verification.1,
            },
            canonical,
            workload,
            evidence,
            hardware: canonical.hardware,
            tensors: canonical.graph.tensors,
            arenas: parts.arenas,
            memory_slots: parts.memory_slots,
            storages: parts.storages,
            execution_contexts: parts.execution_contexts,
            streams: parts.streams,
            events: parts.events,
            nodes: parts.nodes,
            transfers: parts.transfers,
            transfer_batches: parts.transfer_batches,
            synchronization: parts.synchronization,
            schedule: parts.schedule,
            gpu_graph_segments: parts.gpu_graph_segments,
        };

        let plan_hash = plan.compute_payload_hash();
        if (plan_hash.0 | plan_hash.1) == 0 {
            return Err(PlanError::InvalidPlanSeal);
        }
        plan.seal.plan_hash_hi = plan_hash.0;
        plan.seal.plan_hash_lo = plan_hash.1;
        plan.validate()?;
        Ok(plan)
    }

    pub(crate) fn candidate_parts(&self) -> FrozenPlanParts {
        FrozenPlanParts {
            name: self.name,
            arenas: self.arenas,
            memory_slots: self.memory_slots,
            storages: self.storages,
            execution_contexts: self.execution_contexts,
            streams: self.streams,
            events: self.events,
            nodes: self.nodes,
            transfers: self.transfers,
            transfer_batches: self.transfer_batches,
            synchronization: self.synchronization,
            schedule: self.schedule,
            gpu_graph_segments: self.gpu_graph_segments,
        }
    }

    pub(crate) fn compute_payload_hash(&self) -> (u64, u64) {
        let mut hi = StableIdentityHasher::new(0x5541_5344_4541_504c); // UASDEAPL
        let mut lo = StableIdentityHasher::new(0x4652_4f5a_454e_504c); // FROZENPL

        macro_rules! hash_field {
            ($value:expr) => {{
                $value.hash(&mut hi);
                $value.hash(&mut lo);
            }};
        }

        hash_field!(self.name);
        hash_field!(self.key);
        hash_field!(self.canonical);
        hash_field!(self.workload);
        hash_field!(self.evidence);
        hash_field!(self.hardware);
        hash_field!(self.tensors);
        hash_field!(self.arenas);
        hash_field!(self.memory_slots);
        hash_field!(self.storages);
        hash_field!(self.execution_contexts);
        hash_field!(self.streams);
        hash_field!(self.events);
        hash_field!(self.nodes);
        hash_field!(self.transfers);
        hash_field!(self.transfer_batches);
        hash_field!(self.synchronization);
        hash_field!(self.schedule);
        hash_field!(self.gpu_graph_segments);

        (hi.finish(), lo.finish())
    }

    #[inline(always)]
    pub const fn name(&self) -> &'static str {
        self.name
    }
    #[inline(always)]
    pub const fn key(&self) -> PlanKey {
        self.key
    }
    #[inline(always)]
    pub const fn seal(&self) -> PlanSeal {
        self.seal
    }
    #[inline(always)]
    pub const fn canonical_workload(&self) -> CanonicalWorkload {
        self.canonical
    }
    #[inline(always)]
    pub const fn workload(&self) -> WorkloadSignature {
        self.workload
    }
    #[inline(always)]
    pub const fn selection_evidence(&self) -> SelectionEvidence {
        self.evidence
    }
    #[inline(always)]
    pub const fn hardware(&self) -> &HardwareContract {
        &self.hardware
    }
    #[inline(always)]
    pub const fn tensors(&self) -> &'static [TensorSpec] {
        self.tensors
    }
    #[inline(always)]
    pub const fn arenas(&self) -> &'static [MemoryArenaPlan] {
        self.arenas
    }
    #[inline(always)]
    pub const fn memory_slots(&self) -> &'static [MemorySlotPlan] {
        self.memory_slots
    }
    #[inline(always)]
    pub const fn storages(&self) -> &'static [TensorStoragePlan] {
        self.storages
    }
    #[inline(always)]
    pub const fn gpu_graph_segments(&self) -> &'static [GpuGraphSegmentPlan] {
        self.gpu_graph_segments
    }
    #[inline(always)]
    pub const fn execution_contexts(&self) -> &'static [GpuExecutionContextPlan] {
        self.execution_contexts
    }
    #[inline(always)]
    pub const fn streams(&self) -> &'static [StreamPlan] {
        self.streams
    }
    #[inline(always)]
    pub const fn events(&self) -> &'static [EventPlan] {
        self.events
    }
    #[inline(always)]
    pub const fn nodes(&self) -> &'static [FrozenNode] {
        self.nodes
    }
    #[inline(always)]
    pub const fn transfers(&self) -> &'static [TransferPlan] {
        self.transfers
    }
    #[inline(always)]
    pub const fn transfer_batches(&self) -> &'static [TransferBatchPlan] {
        self.transfer_batches
    }
    #[inline(always)]
    pub const fn synchronization(&self) -> &'static [SyncPlan] {
        self.synchronization
    }
    #[inline(always)]
    pub const fn schedule(&self) -> &'static [ScheduleItem] {
        self.schedule
    }
}
