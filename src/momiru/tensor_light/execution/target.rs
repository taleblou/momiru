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
// Subcomponent:  Target
// Module:        momiru::tensor_light::execution::target
// Source:        src/tensor_light/execution/target.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Hardware target capabilities and native artifact metadata for UASDEA plan
//   admission.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Define exact target hardware descriptors and capability requirements.
//   - Describe compatible native artifacts without claiming device execution.
//   - Support validation of target and artifact compatibility.
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

//! Target hardware contract and target-native artifact descriptors.

use super::{stable_hash_pair, MemorySlotId, PlanError};

// =============================================================================
// Hardware capability contracts
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FeatureMask(pub u128);

impl FeatureMask {
    #[inline(always)]
    pub const fn contains(self, required: Self) -> bool {
        (self.0 & required.0) == required.0
    }

    #[inline(always)]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

pub mod cpu_feature {
    use super::FeatureMask;

    pub const SSE42: FeatureMask = FeatureMask(1 << 0);
    pub const AVX: FeatureMask = FeatureMask(1 << 1);
    pub const AVX2: FeatureMask = FeatureMask(1 << 2);
    pub const FMA: FeatureMask = FeatureMask(1 << 3);
    pub const AVX_VNNI: FeatureMask = FeatureMask(1 << 4);
    pub const AVX512F: FeatureMask = FeatureMask(1 << 5);
    pub const AVX512BW: FeatureMask = FeatureMask(1 << 6);
    pub const AVX512VL: FeatureMask = FeatureMask(1 << 7);
    pub const AVX512_VNNI: FeatureMask = FeatureMask(1 << 8);
    pub const AVX512_BF16: FeatureMask = FeatureMask(1 << 9);
    pub const AVX512_FP16: FeatureMask = FeatureMask(1 << 10);
    pub const AMX_TILE: FeatureMask = FeatureMask(1 << 11);
    pub const AMX_INT8: FeatureMask = FeatureMask(1 << 12);
    pub const AMX_BF16: FeatureMask = FeatureMask(1 << 13);
    pub const AMX_FP16: FeatureMask = FeatureMask(1 << 14);
    pub const NEON: FeatureMask = FeatureMask(1 << 15);
    pub const SVE: FeatureMask = FeatureMask(1 << 16);
    pub const SVE2: FeatureMask = FeatureMask(1 << 17);
    pub const SME: FeatureMask = FeatureMask(1 << 18);
    pub const SME2: FeatureMask = FeatureMask(1 << 19);
    pub const RVV: FeatureMask = FeatureMask(1 << 20);
    pub const F16C: FeatureMask = FeatureMask(1 << 21);
    pub const ARM_DOTPROD: FeatureMask = FeatureMask(1 << 22);
    pub const ARM_I8MM: FeatureMask = FeatureMask(1 << 23);
    pub const ARM_BF16: FeatureMask = FeatureMask(1 << 24);
    pub const ARM_FP16: FeatureMask = FeatureMask(1 << 25);
    pub const SVE_BF16: FeatureMask = FeatureMask(1 << 26);
    pub const SVE_I8MM: FeatureMask = FeatureMask(1 << 27);
    pub const SVE_F32MM: FeatureMask = FeatureMask(1 << 28);
    pub const SVE_F64MM: FeatureMask = FeatureMask(1 << 29);
    pub const AVX_IFMA: FeatureMask = FeatureMask(1 << 30);
    pub const AVX_NE_CONVERT: FeatureMask = FeatureMask(1 << 31);
    pub const AVX_VNNI_INT8: FeatureMask = FeatureMask(1 << 32);
    pub const AVX_VNNI_INT16: FeatureMask = FeatureMask(1 << 33);
    pub const AMX_COMPLEX: FeatureMask = FeatureMask(1 << 34);
    pub const AMX_AVX512: FeatureMask = FeatureMask(1 << 35);
    pub const AMX_FP8: FeatureMask = FeatureMask(1 << 36);
    pub const AMX_MOVRS: FeatureMask = FeatureMask(1 << 37);
    pub const AMX_TF32: FeatureMask = FeatureMask(1 << 38);
    /// Reserved for toolchains that expose AVX10.1 target support.
    pub const AVX10_1: FeatureMask = FeatureMask(1 << 39);
    pub const AVX10_2: FeatureMask = FeatureMask(1 << 40);
    pub const APX_F: FeatureMask = FeatureMask(1 << 41);
    pub const MOVRS: FeatureMask = FeatureMask(1 << 42);
    pub const BMI1: FeatureMask = FeatureMask(1 << 43);
    pub const BMI2: FeatureMask = FeatureMask(1 << 44);
    pub const POPCNT: FeatureMask = FeatureMask(1 << 45);
    pub const LZCNT: FeatureMask = FeatureMask(1 << 46);
    pub const AVX512DQ: FeatureMask = FeatureMask(1 << 47);
    pub const AVX512CD: FeatureMask = FeatureMask(1 << 48);
    pub const AVX512IFMA: FeatureMask = FeatureMask(1 << 49);
    pub const AVX512VBMI: FeatureMask = FeatureMask(1 << 50);
    pub const AVX512VBMI2: FeatureMask = FeatureMask(1 << 51);
    pub const AVX512BITALG: FeatureMask = FeatureMask(1 << 52);
    pub const AVX512VPOPCNTDQ: FeatureMask = FeatureMask(1 << 53);
    pub const GFNI: FeatureMask = FeatureMask(1 << 54);
    pub const VAES: FeatureMask = FeatureMask(1 << 55);
    pub const VPCLMULQDQ: FeatureMask = FeatureMask(1 << 56);
    pub const PREFETCHWT1: FeatureMask = FeatureMask(1 << 57);
}

pub mod gpu_feature {
    use super::FeatureMask;

    pub const TENSOR_CORES: FeatureMask = FeatureMask(1 << 0);
    pub const TF32: FeatureMask = FeatureMask(1 << 1);
    pub const FP16: FeatureMask = FeatureMask(1 << 2);
    pub const BF16: FeatureMask = FeatureMask(1 << 3);
    pub const FP8: FeatureMask = FeatureMask(1 << 4);
    pub const FP4: FeatureMask = FeatureMask(1 << 5);
    pub const INT8: FeatureMask = FeatureMask(1 << 6);
    pub const CP_ASYNC: FeatureMask = FeatureMask(1 << 7);
    pub const TMA: FeatureMask = FeatureMask(1 << 8);
    pub const TMA_MULTICAST: FeatureMask = FeatureMask(1 << 9);
    pub const ASYNC_BARRIER: FeatureMask = FeatureMask(1 << 10);
    pub const WARP_SPECIALIZATION: FeatureMask = FeatureMask(1 << 11);
    pub const THREAD_BLOCK_CLUSTERS: FeatureMask = FeatureMask(1 << 12);
    pub const DISTRIBUTED_SHARED_MEMORY: FeatureMask = FeatureMask(1 << 13);
    pub const WGMMA: FeatureMask = FeatureMask(1 << 14);
    pub const TCGEN05: FeatureMask = FeatureMask(1 << 15);
    pub const BLOCK_SCALED_MMA: FeatureMask = FeatureMask(1 << 16);
    pub const CUDA_GRAPH: FeatureMask = FeatureMask(1 << 17);
    pub const DEVICE_GRAPH_LAUNCH: FeatureMask = FeatureMask(1 << 18);
    pub const PROGRAMMATIC_DEPENDENT_LAUNCH: FeatureMask = FeatureMask(1 << 19);
    pub const ASYNC_MEMORY_POOL: FeatureMask = FeatureMask(1 << 20);
    pub const PERSISTING_L2: FeatureMask = FeatureMask(1 << 21);
    pub const MAPPED_HOST_MEMORY: FeatureMask = FeatureMask(1 << 22);
    pub const UNIFIED_MEMORY: FeatureMask = FeatureMask(1 << 23);
    pub const PEER_TO_PEER: FeatureMask = FeatureMask(1 << 24);
    pub const COOPERATIVE_LAUNCH: FeatureMask = FeatureMask(1 << 25);
    pub const GREEN_CONTEXT: FeatureMask = FeatureMask(1 << 26);
    pub const LOCALITY_DOMAINS: FeatureMask = FeatureMask(1 << 27);
    pub const STREAM_PRIORITIES: FeatureMask = FeatureMask(1 << 28);
    pub const MEMORY_SYNC_DOMAINS: FeatureMask = FeatureMask(1 << 29);
    pub const NVLINK_UTIL_CENTRIC_SCHEDULING: FeatureMask = FeatureMask(1 << 30);
    pub const BATCHED_ASYNC_COPY: FeatureMask = FeatureMask(1 << 31);
    pub const PREFERRED_CLUSTER_DIMENSION: FeatureMask = FeatureMask(1 << 32);
    pub const NON_PORTABLE_CLUSTER_SIZE: FeatureMask = FeatureMask(1 << 33);
    pub const NON_PORTABLE_SHARED_MEMORY_MODE: FeatureMask = FeatureMask(1 << 34);
    pub const EXECUTION_AFFINITY: FeatureMask = FeatureMask(1 << 35);
    pub const FP6: FeatureMask = FeatureMask(1 << 36);
    pub const TENSOR_MEMORY: FeatureMask = FeatureMask(1 << 37);
    /// Green-context workqueue resource configuration when exposed by the selected CUDA toolchain.
    pub const GREEN_CONTEXT_WORKQUEUE: FeatureMask = FeatureMask(1 << 38);
    /// Hopper+ dynamic-programming acceleration instructions.
    pub const DPX: FeatureMask = FeatureMask(1 << 39);
    /// Native FP64 matrix/tensor acceleration when exposed by the target.
    pub const FP64_MATRIX_ACCELERATION: FeatureMask = FeatureMask(1 << 40);
    /// Blackwell+ work stealing using Cluster Launch Control.
    pub const CLUSTER_LAUNCH_CONTROL: FeatureMask = FeatureMask(1 << 41);
    /// CUDA Virtual Memory Management for fine-grained mapped allocations.
    pub const VIRTUAL_MEMORY_MANAGEMENT: FeatureMask = FeatureMask(1 << 42);
    /// NVLink/NVSwitch fabric-addressable memory mappings.
    pub const FABRIC_MEMORY: FeatureMask = FeatureMask(1 << 43);
    /// Extended GPU Memory on supported CPU-GPU NVLink-C2C platforms.
    pub const EXTENDED_GPU_MEMORY: FeatureMask = FeatureMask(1 << 44);
    /// Small-scale asynchronous register-to-distributed-shared-memory transfers.
    pub const STAS: FeatureMask = FeatureMask(1 << 45);
    /// Per-launch shared-memory mode selection on capable CUDA targets.
    pub const SHARED_MEMORY_MODE: FeatureMask = FeatureMask(1 << 46);
    /// Oversized shared-memory launch support on capable CUDA targets.
    pub const OVERSIZED_SHARED_MEMORY: FeatureMask = FeatureMask(1 << 47);
    /// Per-kernel launch priority attribute.
    pub const KERNEL_LAUNCH_PRIORITY: FeatureMask = FeatureMask(1 << 48);
    /// Launch-completion events exposed by extended launch attributes.
    pub const LAUNCH_COMPLETION_EVENT: FeatureMask = FeatureMask(1 << 49);
    /// CUDA VMM multicast objects / NVLink SHARP capable mappings.
    pub const MULTICAST_MEMORY: FeatureMask = FeatureMask(1 << 50);
    /// CUDA graph kernel nodes whose launch parameters may be updated from device code.
    pub const DEVICE_UPDATABLE_GRAPH_NODE: FeatureMask = FeatureMask(1 << 51);
    /// Target-native structured-sparse matrix/tensor acceleration (mma.sp/tcgen05.mma.sp).
    pub const STRUCTURED_SPARSE_MMA: FeatureMask = FeatureMask(1 << 52);
    /// Device can directly access pageable host memory.
    pub const PAGEABLE_HOST_MEMORY_ACCESS: FeatureMask = FeatureMask(1 << 53);
    /// Pageable host memory is accessed through host page tables, allowing a
    /// true no-registration coherent host allocation candidate on supported systems.
    pub const PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES: FeatureMask = FeatureMask(1 << 54);
    /// Host can directly access managed allocations without an explicit migration API.
    pub const DIRECT_MANAGED_MEMORY_ACCESS_FROM_HOST: FeatureMask = FeatureMask(1 << 55);
    /// Native atomics between the device and host memory are supported.
    pub const HOST_NATIVE_ATOMICS: FeatureMask = FeatureMask(1 << 56);
    /// Registered host pointers may be used directly as device pointers.
    pub const CAN_USE_HOST_POINTER_FOR_REGISTERED_MEMORY: FeatureMask = FeatureMask(1 << 57);
    /// Read-only host registration is supported by the CUDA runtime/device pair.
    pub const HOST_REGISTER_READ_ONLY: FeatureMask = FeatureMask(1 << 58);
    /// Managed allocations may be coherently accessed by the GPU concurrently
    /// with CPU access. Required before candidate generation may assume true
    /// simultaneous CPU/GPU Unified-Memory access without ownership handoff.
    pub const CONCURRENT_MANAGED_ACCESS: FeatureMask = FeatureMask(1 << 59);
    /// Host-memory registration is supported by the device/runtime pair. This is
    /// separate from cudaHostAlloc-style pinned allocations and matters for
    /// zero-copy reuse of externally-owned host buffers.
    pub const HOST_REGISTER_SUPPORTED: FeatureMask = FeatureMask(1 << 60);
    /// CUDA runtime/toolchain supports batched managed-memory prefetch and the
    /// probed system-wide GPU set satisfies the concurrent-managed-access
    /// requirement of cudaMemPrefetchBatchAsync.
    pub const BATCHED_MANAGED_PREFETCH: FeatureMask = FeatureMask(1 << 61);
    /// CUDA driver/toolchain supports batched managed/system-memory discard and
    /// discard+prefetch operations; candidate generation must still require the
    /// system-wide concurrent-managed-access contract mandated by CUDA.
    pub const BATCHED_MEMORY_DISCARD: FeatureMask = FeatureMask(1 << 62);
    /// Hopper+ one-dimensional cp.async.bulk global/shared transfers.
    pub const BULK_ASYNC_COPY: FeatureMask = FeatureMask(1 << 63);
    /// Tensor-map im2col descriptors (CC 9.0+).
    pub const TENSOR_MAP_IM2COL: FeatureMask = FeatureMask(1 << 64);
    /// Wide im2col tensor maps (CC 10.0+).
    pub const TENSOR_MAP_IM2COL_WIDE: FeatureMask = FeatureMask(1 << 65);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuArchitecture {
    X86_64,
    AArch64,
    RiscV64,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuLogicalProcessor {
    pub processor_group: u16,
    pub logical_cpu_id: u16,
    pub physical_core_id: u16,
    pub numa_node: u16,
    pub package_id: u16,
    /// Relative hardware efficiency/performance class. Exact interpretation is
    /// platform-specific; offline planning uses the frozen profile values.
    pub efficiency_class: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuHardwareContract {
    pub architecture: CpuArchitecture,
    pub features: FeatureMask,
    pub logical_cpu_count: u16,
    pub physical_core_count: u16,
    pub processors: &'static [CpuLogicalProcessor],
    pub numa_node_count: u16,
    pub cache_line_bytes: u16,
    pub l1_data_bytes_per_core: u32,
    pub l2_bytes_per_core: u32,
    pub l3_bytes_shared: u64,
    pub max_vector_bytes: u16,
    pub sve_vector_bytes: u16,
    pub sme_vector_bytes: u16,
    /// Exact RISC-V VLEN in bytes when RVV is available; zero otherwise.
    pub rvv_vector_bytes: u16,
}

impl CpuHardwareContract {
    /// Check that the frozen CPU inventory identifies exactly the claimed
    /// number of *distinct* physical cores and unique logical processors.
    /// The core IDs are canonical dense indices 0..physical_core_count; a
    /// package-local core ID must be translated to a distinct global ID by
    /// the OFFLINE hardware enumerator before this contract is constructed.
    ///
    /// This bounded, allocation-free check runs at profile validation or
    /// offline candidate seeding, never in a compiled tensor kernel. It does
    /// not establish the runtime OS-enforced affinity or ISA eligibility proof.
    pub fn has_consistent_processor_inventory(&self) -> bool {
        if self.logical_cpu_count == 0
            || self.physical_core_count == 0
            || self.physical_core_count > self.logical_cpu_count
            || self.numa_node_count == 0
            || self.processors.len() != usize::from(self.logical_cpu_count)
        {
            return false;
        }

        // A u16 physical core ID fits in exactly 65536 bits (8 KiB).
        // This bound avoids allocating a Vec during a profile safety check.
        let mut seen_physical = [0u64; 1024];
        let mut distinct_physical = 0usize;
        let mut previous: Option<(u16, u16)> = None;
        for processor in self.processors {
            if processor.numa_node >= self.numa_node_count
                || processor.physical_core_id >= self.physical_core_count
            {
                return false;
            }
            let logical_id = (processor.processor_group, processor.logical_cpu_id);
            if previous.is_some_and(|last| last >= logical_id) {
                return false;
            }
            previous = Some(logical_id);

            let physical_id = usize::from(processor.physical_core_id);
            let word = physical_id / 64;
            let bit = 1u64 << (physical_id % 64);
            if seen_physical[word] & bit == 0 {
                seen_physical[word] |= bit;
                distinct_physical += 1;
            }
        }
        distinct_physical == usize::from(self.physical_core_count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuHardwareContract {
    pub features: FeatureMask,
    pub device_ordinal: u16,
    /// Exact native target label, e.g. sm_89, sm_90a, sm_100a.
    pub native_arch: &'static str,
    pub compute_major: u8,
    pub compute_minor: u8,
    pub sm_count: u16,
    pub min_sm_partition_size: u16,
    pub sm_coscheduled_alignment: u16,
    pub warp_size: u8,
    pub max_threads_per_block: u16,
    pub max_threads_per_sm: u16,
    pub max_warps_per_sm: u16,
    pub max_blocks_per_sm: u8,
    pub max_registers_per_thread: u16,
    pub max_registers_per_block: u32,
    pub registers_per_sm: u32,
    pub max_shared_bytes_per_block: u32,
    /// Maximum dynamic shared memory under the normal opt-in limit.
    pub max_dynamic_shared_bytes_per_block: u32,
    /// Portable per-block dynamic shared-memory limit.
    pub max_portable_dynamic_shared_bytes_per_block: u32,
    /// Oversized dynamic shared-memory limit; 0 means unsupported.
    pub max_oversized_dynamic_shared_bytes_per_block: u32,
    pub max_shared_bytes_per_sm: u32,
    /// Maximum CUDA access-policy window accepted by this device.
    pub max_access_policy_window_bytes: usize,
    /// Maximum amount of L2 that may be reserved for persisting accesses.
    pub max_persisting_l2_bytes: usize,
    /// Full L2 capacity used by the offline locality/cache cost model.
    pub l2_cache_bytes: usize,
    /// Total device constant-memory capacity available to __constant__ symbols.
    pub constant_memory_bytes: usize,
    /// Peak external-memory interface width and clock reported by the device.
    pub memory_bus_width_bits: u32,
    pub memory_clock_khz: u32,
    /// Number of asynchronous copy engines and whether independent kernels may overlap.
    pub async_engine_count: u8,
    pub concurrent_kernels: bool,
    /// Host NUMA node nearest to the GPU, when the platform reports one.
    pub host_numa_node: Option<u16>,
    pub max_block_dim: [u32; 3],
    pub max_grid_dim: [u32; 3],
    pub locality_domain_count: u16,
    pub memory_sync_domain_count: u8,
    /// Highest-priority stream value reported by CUDA (numerically smallest).
    pub greatest_stream_priority: i32,
    /// Lowest-priority stream value reported by CUDA (numerically largest).
    pub least_stream_priority: i32,
    /// Maximum workqueue concurrency hint reported by the device/context.
    pub max_workqueue_concurrency: u16,
    pub max_portable_cluster_blocks: u16,
    pub max_tmem_columns_per_cta: u16,
    pub tmem_allocation_granularity: u16,
    /// CUDA multicast allocation granularity. Zero means multicast unsupported.
    pub multicast_min_granularity_bytes: usize,
    pub multicast_recommended_granularity_bytes: usize,
    /// Optional architecture-specific exclusive TMEM allocation (e.g. newer
    /// targets may expose a non-power-of-two exclusive allocation size).
    pub tmem_exclusive_columns: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HardwareContract {
    pub profile_hash_hi: u64,
    pub profile_hash_lo: u64,
    pub cpu: Option<CpuHardwareContract>,
    pub gpus: &'static [GpuHardwareContract],
}

/// Stable digest of the actual hardware payload, excluding the digest fields
/// themselves. The device/processor slices must already be in canonical order.
pub fn canonical_hardware_profile_hash(
    cpu: Option<CpuHardwareContract>,
    gpus: &[GpuHardwareContract],
) -> (u64, u64) {
    stable_hash_pair(&(cpu, gpus), 0x4841_5244_5741_5245) // HARDWARE
}

impl HardwareContract {
    /// Construct a contract whose identity is mechanically derived from the
    /// exact CPU/GPU capability payload.
    pub fn canonical(
        cpu: Option<CpuHardwareContract>,
        gpus: &'static [GpuHardwareContract],
    ) -> Self {
        let hash = canonical_hardware_profile_hash(cpu, gpus);
        Self {
            profile_hash_hi: hash.0,
            profile_hash_lo: hash.1,
            cpu,
            gpus,
        }
    }

    #[inline]
    pub fn computed_profile_hash(&self) -> (u64, u64) {
        canonical_hardware_profile_hash(self.cpu, self.gpus)
    }

    /// Validate the canonical hardware payload independently of any selected
    /// execution plan. This runs before offline candidate generation.
    pub fn validate_canonical(&self) -> Result<(), PlanError> {
        let computed = self.computed_profile_hash();
        if (computed.0 | computed.1) == 0
            || self.profile_hash_hi != computed.0
            || self.profile_hash_lo != computed.1
        {
            return Err(PlanError::InvalidHardwareContract);
        }

        if let Some(cpu) = self.cpu {
            if cpu.logical_cpu_count == 0
                || cpu.physical_core_count == 0
                || cpu.physical_core_count > cpu.logical_cpu_count
                || cpu.numa_node_count == 0
                || cpu.cache_line_bytes == 0
                || !cpu.cache_line_bytes.is_power_of_two()
                || (cpu.max_vector_bytes > 0 && cpu.max_vector_bytes % 16 != 0)
                || (cpu.sve_vector_bytes > 0
                    && (cpu.sve_vector_bytes % 16 != 0
                        || cpu.sve_vector_bytes > cpu.max_vector_bytes))
                || (cpu.sme_vector_bytes > 0
                    && (cpu.sme_vector_bytes % 16 != 0
                        || cpu.sme_vector_bytes > cpu.max_vector_bytes))
                || (cpu.rvv_vector_bytes > 0
                    && (cpu.rvv_vector_bytes % 8 != 0
                        || cpu.rvv_vector_bytes > cpu.max_vector_bytes))
                || !cpu.has_consistent_processor_inventory()
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if cpu.features.contains(cpu_feature::SVE) && cpu.sve_vector_bytes == 0 {
                return Err(PlanError::InvalidHardwareContract);
            }
            if cpu.features.contains(cpu_feature::SME) && cpu.sme_vector_bytes == 0 {
                return Err(PlanError::InvalidHardwareContract);
            }
            if cpu.features.contains(cpu_feature::RVV) && cpu.rvv_vector_bytes == 0 {
                return Err(PlanError::InvalidHardwareContract);
            }
            if !cpu.features.contains(cpu_feature::RVV) && cpu.rvv_vector_bytes != 0 {
                return Err(PlanError::InvalidHardwareContract);
            }
            let mut i = 0usize;
            while i < cpu.processors.len() {
                let processor = &cpu.processors[i];
                if processor.numa_node >= cpu.numa_node_count
                    || processor.physical_core_id >= cpu.physical_core_count
                {
                    return Err(PlanError::InvalidHardwareContract);
                }
                if i > 0 {
                    let previous = &cpu.processors[i - 1];
                    if (previous.processor_group, previous.logical_cpu_id)
                        >= (processor.processor_group, processor.logical_cpu_id)
                    {
                        return Err(PlanError::InvalidHardwareContract);
                    }
                }
                i += 1;
            }
        }

        let mut i = 0usize;
        while i < self.gpus.len() {
            let gpu = &self.gpus[i];
            if gpu.native_arch.is_empty()
                || gpu.compute_major == 0
                || gpu.sm_count == 0
                || gpu.warp_size == 0
                || gpu.max_threads_per_block == 0
                || gpu.max_threads_per_sm == 0
                || gpu.max_warps_per_sm == 0
                || gpu.max_blocks_per_sm == 0
                || gpu.max_registers_per_thread == 0
                || gpu.max_registers_per_block == 0
                || gpu.registers_per_sm == 0
                || gpu.max_registers_per_block > gpu.registers_per_sm
                || gpu.max_shared_bytes_per_block == 0
                || gpu.max_shared_bytes_per_block > gpu.max_shared_bytes_per_sm
                || gpu.max_dynamic_shared_bytes_per_block == 0
                || gpu.max_portable_dynamic_shared_bytes_per_block == 0
                || gpu.max_portable_dynamic_shared_bytes_per_block
                    > gpu.max_dynamic_shared_bytes_per_block
                || (gpu.max_oversized_dynamic_shared_bytes_per_block > 0
                    && gpu.max_oversized_dynamic_shared_bytes_per_block
                        < gpu.max_dynamic_shared_bytes_per_block)
                || gpu.max_shared_bytes_per_sm == 0
                || gpu.max_block_dim.iter().any(|&v| v == 0)
                || gpu.max_grid_dim.iter().any(|&v| v == 0)
                || gpu.locality_domain_count == 0
                || gpu.memory_sync_domain_count == 0
                || gpu.greatest_stream_priority > gpu.least_stream_priority
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES)
                && !gpu.features.contains(gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::DIRECT_MANAGED_MEMORY_ACCESS_FROM_HOST)
                && !gpu.features.contains(gpu_feature::UNIFIED_MEMORY)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::CONCURRENT_MANAGED_ACCESS)
                && !gpu.features.contains(gpu_feature::UNIFIED_MEMORY)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::HOST_REGISTER_READ_ONLY)
                && !gpu.features.contains(gpu_feature::HOST_REGISTER_SUPPORTED)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::CAN_USE_HOST_POINTER_FOR_REGISTERED_MEMORY)
                && !gpu.features.contains(gpu_feature::HOST_REGISTER_SUPPORTED)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::PROGRAMMATIC_DEPENDENT_LAUNCH)
                && gpu.compute_major < 9
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::BATCHED_MANAGED_PREFETCH)
                && !gpu.features.contains(gpu_feature::CONCURRENT_MANAGED_ACCESS)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if let (Some(host_node), Some(cpu)) = (gpu.host_numa_node, self.cpu) {
                if host_node >= cpu.numa_node_count {
                    return Err(PlanError::InvalidHardwareContract);
                }
            }
            if gpu.features.contains(gpu_feature::GREEN_CONTEXT)
                && (gpu.min_sm_partition_size == 0 || gpu.sm_coscheduled_alignment == 0)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::PERSISTING_L2)
                && (gpu.max_access_policy_window_bytes == 0
                    || gpu.max_persisting_l2_bytes == 0
                    || gpu.l2_cache_bytes == 0
                    || gpu.max_persisting_l2_bytes > gpu.l2_cache_bytes)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::TENSOR_MEMORY)
                && (gpu.max_tmem_columns_per_cta == 0 || gpu.tmem_allocation_granularity == 0)
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::MULTICAST_MEMORY) {
                if !gpu.features.contains(gpu_feature::VIRTUAL_MEMORY_MANAGEMENT)
                    || gpu.multicast_min_granularity_bytes == 0
                    || gpu.multicast_recommended_granularity_bytes
                        < gpu.multicast_min_granularity_bytes
                    || gpu.multicast_recommended_granularity_bytes
                        % gpu.multicast_min_granularity_bytes
                        != 0
                {
                    return Err(PlanError::InvalidHardwareContract);
                }
            } else if gpu.multicast_min_granularity_bytes != 0
                || gpu.multicast_recommended_granularity_bytes != 0
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if gpu.features.contains(gpu_feature::GREEN_CONTEXT_WORKQUEUE)
                && gpu.max_workqueue_concurrency == 0
            {
                return Err(PlanError::InvalidHardwareContract);
            }
            if i > 0 {
                let previous = self.gpus[i - 1].device_ordinal;
                if previous == gpu.device_ordinal {
                    return Err(PlanError::DuplicateGpuOrdinal);
                }
                if previous > gpu.device_ordinal {
                    return Err(PlanError::InvalidHardwareContract);
                }
            }
            i += 1;
        }
        Ok(())
    }
}



// =============================================================================
// Unified target-native artifact reference
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArtifactKind {
    RustAotFunction,
    NativeObject,
    StaticLibrary,
    /// Exact target-native NVIDIA machine image. PTX/fatbin/JIT inputs may be
    /// used by the offline toolchain but are never executable frozen artifacts.
    Cubin,
    /// Exact vendor-library kernel/algorithm binding with heuristics disabled.
    LibraryKernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub artifact: &'static str,
    pub entry: &'static str,
    pub artifact_hash_hi: u64,
    pub artifact_hash_lo: u64,
    /// Exact hardware-profile digest this artifact was selected and verified for.
    /// Binding rejects the artifact when it does not match the frozen plan profile.
    pub target_profile_hash_hi: u64,
    pub target_profile_hash_lo: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LibraryProvider {
    None,
    Cublas,
    CublasLt,
    CuFft,
    CuSolver,
    Cudnn,
    CuTensor,
    CuSparse,
    Nccl,
    OneDnn,
    Mkl,
    OpenBlas,
    Accelerate,
    Blis,
    /// Generic BLAS provider when a more precise provider identity is unavailable.
    Blas,
    TargetNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LibraryKnob {
    pub key: u32,
    pub value: i64,
}

/// Exact CPU placement for a provider-managed library worker thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProviderThreadBinding {
    pub processor_group: u16,
    pub logical_cpu_id: u16,
    pub numa_node: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LibraryAlgorithmPlan {
    pub provider: LibraryProvider,
    pub provider_version: &'static str,
    pub algorithm_id: i64,
    pub workspace_bytes: usize,
    pub workspace_slot: Option<MemorySlotId>,
    pub knobs: &'static [LibraryKnob],
    /// Exact provider-managed CPU thread count selected offline. Zero means
    /// the provider has no provider-managed CPU worker pool (typical GPU libraries).
    pub provider_thread_count: u16,
    /// Exact processor bindings for provider-managed CPU threads. Empty when
    /// `provider_thread_count == 0` (typical GPU libraries).
    pub provider_thread_bindings: &'static [ProviderThreadBinding],
    /// True only when provider-side dynamic thread-count selection is disabled
    /// and `provider_thread_count` is deterministically enforced at binding/replay.
    pub provider_threading_fixed: bool,
    /// Digest of the exact provider configuration/descriptor selected offline.
    pub configuration_hash_hi: u64,
    pub configuration_hash_lo: u64,
    /// These must be true for a frozen library artifact: runtime heuristic
    /// selection and provider-side autotuning are not allowed during replay.
    pub exact_algorithm_binding: bool,
    pub runtime_heuristics_disabled: bool,
    pub runtime_autotune_disabled: bool,
}

