// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{select_screening_winner, ScreeningRank};
use crate::tensor_light::execution::CandidateId;

// These integers exercise the ordering routine only. They are not benchmark
// observations or evidence used to freeze a native plan.
fn rank(id: u64, median: u64, uncertainty: u64, workspace: usize) -> ScreeningRank {
    ScreeningRank {
        id: CandidateId { hi: 1, lo: id },
        median_ns: median,
        p95_ns: median.saturating_add(2),
        uncertainty_ns: uncertainty,
        mad_ns: 0,
        workspace_bytes: workspace,
        estimated_shared_memory_bytes: 0,
        estimated_register_bytes: 0,
    }
}

#[test]
fn overlapping_chain_does_not_make_a_dominated_candidate_eligible() {
    let ranks = [
        rank(1, 100, 10, 64),
        rank(2, 110, 14, 16),
        rank(3, 125, 10, 0),
    ];
    // A overlaps B and B overlaps C, but C's lower bound is above A's
    // upper bound. B wins on workspace and A remains the fastest challenger.
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let permuted = order.map(|i| ranks[i]);
        let (winner, challenger) = select_screening_winner(&permuted);
        assert_eq!(permuted[winner].id, ranks[1].id);
        assert_eq!(challenger.map(|i| permuted[i].id), Some(ranks[0].id));
    }
}

#[test]
fn touching_intervals_stay_eligible_and_exact_ties_use_candidate_identity() {
    let ranks = [
        rank(2, 120, 10, 8),
        rank(1, 120, 10, 8),
        rank(3, 100, 10, 16),
    ];
    let (winner, challenger) = select_screening_winner(&ranks);
    assert_eq!(ranks[winner].id.lo, 1);
    assert_eq!(challenger.map(|i| ranks[i].id.lo), Some(3));
}

#[test]
fn saturated_bounds_and_one_candidate_remain_well_defined() {
    let ranks = [rank(1, u64::MAX - 1, 10, 16), rank(2, u64::MAX, 0, 8)];
    assert_eq!(ranks[0].upper_ns(), u64::MAX);
    assert_eq!(rank(3, 1, 5, 0).lower_ns(), 0);
    assert_eq!(select_screening_winner(&ranks), (1, Some(0)));
    assert_eq!(select_screening_winner(&ranks[..1]), (0, None));
}
