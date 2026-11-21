// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{reserve_replay_steps, ReplayStep};
use crate::tensor_light::execution::{PlanError, SyncKind, SyncPlan};

static SYNC: SyncPlan = SyncPlan {
    id: 0,
    kind: SyncKind::CpuBarrier,
    event: None,
    barrier: Some(0),
    producer_node: 0,
    producer_slice: None,
    consumer_node: 1,
    consumer_slice: None,
    stream: None,
};

#[test]
fn schedule_budget_fits_without_reallocation_or_initialized_entries() {
    let budget = 8;
    let mut steps = reserve_replay_steps(budget).expect("small replay allocation");
    assert!(steps.is_empty());
    let original_ptr = steps.as_ptr();
    let original_capacity = steps.capacity();
    assert!(original_capacity >= budget);
    for _ in 0..budget {
        steps.push(ReplayStep::Sync(&SYNC));
    }
    assert_eq!(steps.len(), budget);
    assert_eq!(steps.as_ptr(), original_ptr);
    assert_eq!(steps.capacity(), original_capacity);
}

#[test]
fn impossible_schedule_budget_returns_the_replay_allocation_error() {
    let above_object_size = isize::MAX as usize / size_of::<ReplayStep>() + 1;
    for budget in [above_object_size, usize::MAX] {
        assert!(matches!(
            reserve_replay_steps(budget),
            Err(PlanError::ReplayProgramAllocationFailed)
        ));
    }
}

#[test]
fn empty_schedule_reserves_no_steps_or_capacity() {
    let steps = reserve_replay_steps(0).expect("empty replay allocation");
    assert!(steps.is_empty());
    assert_eq!(steps.capacity(), 0);
}
