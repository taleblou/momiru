// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use crate::tensor_light::aot::{
    cpu_rank_should_receive_specialized_candidates, cpu_should_seed_parallel,
    cpu_vendor_provider_seeds, gpu_advanced_candidate_seeds, CpuOfflineSeedPolicy,
    GpuOfflineSeedPolicy,
};
use crate::tensor_light::execution::{
    gpu_feature, GpuHardwareContract, LibraryProvider, Operation,
};
use crate::tensor_light::MAX_RANK;

// Synthetic input to the static seed policy. It carries no device attestation
// or evidence of compiled code, kernel execution, or performance.
const CUDA: GpuHardwareContract = GpuHardwareContract {
    features: gpu_feature::BULK_ASYNC_COPY.union(gpu_feature::TMA),
    device_ordinal: 0,
    native_arch: "sm_90a",
    compute_major: 9,
    compute_minor: 0,
    sm_count: 16,
    min_sm_partition_size: 1,
    sm_coscheduled_alignment: 1,
    warp_size: 32,
    max_threads_per_block: 1024,
    max_threads_per_sm: 2048,
    max_warps_per_sm: 64,
    max_blocks_per_sm: 16,
    max_registers_per_thread: 255,
    max_registers_per_block: 65_536,
    registers_per_sm: 65_536,
    max_shared_bytes_per_block: 48 * 1024,
    max_dynamic_shared_bytes_per_block: 48 * 1024,
    max_portable_dynamic_shared_bytes_per_block: 48 * 1024,
    max_oversized_dynamic_shared_bytes_per_block: 0,
    max_shared_bytes_per_sm: 96 * 1024,
    max_access_policy_window_bytes: 0,
    max_persisting_l2_bytes: 0,
    l2_cache_bytes: 4 * 1024 * 1024,
    constant_memory_bytes: 64 * 1024,
    memory_bus_width_bits: 256,
    memory_clock_khz: 1_000_000,
    async_engine_count: 1,
    concurrent_kernels: true,
    host_numa_node: None,
    max_block_dim: [1024, 1024, 64],
    max_grid_dim: [2_147_483_647, 65_535, 65_535],
    locality_domain_count: 1,
    memory_sync_domain_count: 1,
    greatest_stream_priority: 0,
    least_stream_priority: 0,
    max_workqueue_concurrency: 0,
    max_portable_cluster_blocks: 1,
    max_tmem_columns_per_cta: 0,
    tmem_allocation_granularity: 0,
    multicast_min_granularity_bytes: 0,
    multicast_recommended_granularity_bytes: 0,
    tmem_exclusive_columns: 0,
};

const MATMUL: Operation = Operation::MatMul {
    transpose_lhs: false,
    transpose_rhs: false,
};

#[test]
fn cuda_only_seeds_require_matching_native_arch_and_warp() {
    let policy = GpuOfflineSeedPolicy::default();
    let seeds = gpu_advanced_candidate_seeds(MATMUL, &CUDA, 16, policy);
    assert!(seeds.bulk_async_copy && seeds.tma_tiled);
    assert!(seeds.constant_memory && seeds.compileiq_codegen);
    assert_eq!(
        seeds.vendor_providers,
        &[LibraryProvider::CublasLt, LibraryProvider::Cublas]
    );

    for hardware in [
        GpuHardwareContract {
            native_arch: "gfx1100",
            ..CUDA
        },
        GpuHardwareContract {
            native_arch: "sm_89",
            ..CUDA
        },
        GpuHardwareContract {
            native_arch: "sm_90a1",
            ..CUDA
        },
        GpuHardwareContract {
            warp_size: 64,
            ..CUDA
        },
    ] {
        let rejected = gpu_advanced_candidate_seeds(MATMUL, &hardware, 16, policy);
        assert!(
            !rejected.bulk_async_copy && !rejected.tma_tiled,
            "{hardware:?}"
        );
        assert!(!rejected.constant_memory && !rejected.compileiq_codegen);
        assert!(rejected.vendor_providers.is_empty());
    }
}

#[test]
fn constant_memory_uses_the_tighter_hardware_or_policy_limit() {
    let hardware = GpuHardwareContract {
        constant_memory_bytes: 32,
        ..CUDA
    };
    for (policy_limit, allowed) in [(16, 16), (64, 32)] {
        let policy = GpuOfflineSeedPolicy {
            constant_memory_seed_max_bytes: policy_limit,
            seed_bulk_async_copy: false,
            seed_tma_tensor_maps: false,
            seed_vendor_libraries: false,
            seed_compileiq_codegen: false,
            ..GpuOfflineSeedPolicy::default()
        };
        let seeds = gpu_advanced_candidate_seeds(MATMUL, &hardware, allowed, policy);
        assert!(seeds.constant_memory);
        assert!(!seeds.bulk_async_copy && !seeds.tma_tiled && !seeds.compileiq_codegen);
        assert!(seeds.vendor_providers.is_empty());
        for unsupported in [0, allowed + 1] {
            assert!(
                !gpu_advanced_candidate_seeds(MATMUL, &hardware, unsupported, policy)
                    .constant_memory
            );
        }
    }
}

#[test]
fn cpu_policy_caps_rank_and_requires_workers_and_nonempty_work() {
    let policy = CpuOfflineSeedPolicy {
        rank_specialization_limit: u8::MAX,
        parallel_min_elements: 8,
        seed_vendor_libraries: false,
        ..CpuOfflineSeedPolicy::default()
    };
    assert!(!cpu_rank_should_receive_specialized_candidates(0, policy));
    assert!(cpu_rank_should_receive_specialized_candidates(
        MAX_RANK, policy
    ));
    assert!(!cpu_rank_should_receive_specialized_candidates(
        MAX_RANK + 1,
        policy
    ));
    assert!(cpu_vendor_provider_seeds(MATMUL, policy).is_empty());
    assert!(cpu_should_seed_parallel(8, 2, policy));
    assert!(!cpu_should_seed_parallel(7, 2, policy));
    assert!(!cpu_should_seed_parallel(8, 1, policy));
    assert!(!cpu_should_seed_parallel(
        8,
        2,
        CpuOfflineSeedPolicy {
            parallel_tasks_per_worker: 0,
            ..policy
        }
    ));
}
