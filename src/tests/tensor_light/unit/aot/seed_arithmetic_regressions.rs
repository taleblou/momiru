// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use crate::tensor_light::aot::{
    cpu_chunk_elements_from_kib_seed, cpu_parallel_task_seed_with_policy,
    gpu_elementwise_blocks_seed, CpuOfflineSeedPolicy, GpuOfflineSeedPolicy,
};

#[test]
fn cpu_task_seed_handles_full_width_counts_without_rounding_overflow() {
    let policy = CpuOfflineSeedPolicy {
        parallel_tasks_per_worker: u8::MAX,
        ..CpuOfflineSeedPolicy::default()
    };
    assert_eq!(
        cpu_parallel_task_seed_with_policy(u64::MAX, u16::MAX, 2, policy),
        u32::from(u16::MAX) * u32::from(u8::MAX)
    );
    assert_eq!(
        cpu_parallel_task_seed_with_policy(u64::MAX, 8, u64::MAX - 1, policy),
        2
    );
    assert_eq!(cpu_parallel_task_seed_with_policy(7, 64, 0, policy), 7);
    assert_eq!(cpu_parallel_task_seed_with_policy(0, 1, 1, policy), 0);
    assert_eq!(cpu_parallel_task_seed_with_policy(9, 0, 1, policy), 0);
}

#[test]
fn chunk_seed_saturates_kib_conversion_and_respects_input_size() {
    assert_eq!(
        cpu_chunk_elements_from_kib_seed(u64::MAX, u64::MAX),
        2_000_000
    );
    assert_eq!(cpu_chunk_elements_from_kib_seed(117, u64::MAX), 117);
    assert_eq!(cpu_chunk_elements_from_kib_seed(u64::MAX, 0), 4_096);
    assert_eq!(cpu_chunk_elements_from_kib_seed(0, u64::MAX), 1);
}

#[test]
fn gpu_block_seed_rounds_up_before_capping_and_disables_empty_launches() {
    let policy = GpuOfflineSeedPolicy {
        max_elementwise_blocks: 7,
        ..GpuOfflineSeedPolicy::default()
    };
    for (elements, threads, expected) in [
        (0, 32, 0),
        (64, 0, 0),
        (1, 32, 1),
        (64, 32, 2),
        (65, 32, 3),
        (u64::MAX, u16::MAX, 7),
    ] {
        assert_eq!(
            gpu_elementwise_blocks_seed(elements, threads, policy),
            expected
        );
    }
    assert_eq!(
        gpu_elementwise_blocks_seed(
            64,
            32,
            GpuOfflineSeedPolicy {
                max_elementwise_blocks: 0,
                ..policy
            }
        ),
        0
    );
}
