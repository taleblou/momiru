// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{duplicate_exact_identity, exact_key_window};

#[test]
fn sorted_window_matches_linear_search_for_complete_digest_pairs() {
    let index = [
        (1, 2, 8),
        (1, 2, 3),
        (1, 2, 7),
        (1, 4, 9),
        (2, 0, 4),
        (2, 2, 1),
        (2, 2, 5),
    ];
    for hi in 0..=3 {
        for lo in 0..=5 {
            let key = (hi, lo);
            let below = index.iter().filter(|&&(h, l, _)| (h, l) < key).count();
            let matching = index.iter().filter(|&&(h, l, _)| (h, l) == key).count();
            assert_eq!(exact_key_window(&index, key), below..below + matching);
        }
    }
    assert_eq!(exact_key_window(&[], (0, u64::MAX)), 0..0);
    assert_eq!(
        exact_key_window(&[(u64::MAX, u64::MAX, 0)], (u64::MAX, u64::MAX)),
        0..1
    );
}

#[test]
fn equal_hashes_do_not_hide_distinct_identities_or_nonadjacent_duplicates() {
    let index = [(1, 2, 2), (1, 2, 0), (1, 2, 1), (3, 4, 3)];
    // Distinct canonical labels within the bucket and the same label in
    // a different bucket. Only same-key pairs should reach the predicate.
    let distinct = ["A", "B", "C", "C"];
    let mut compared = Vec::new();
    assert!(!duplicate_exact_identity(&index, |a, b| {
        compared.push((a.min(b), a.max(b)));
        distinct[a] == distinct[b]
    }));
    compared.sort_unstable();
    assert_eq!(compared, [(0, 1), (0, 2), (1, 2)]);

    let duplicate = ["A", "C", "C", "D"];
    assert!(duplicate_exact_identity(&index, |a, b| duplicate[a] == duplicate[b]));
}

#[test]
fn empty_and_singleton_buckets_never_request_an_identity_comparison() {
    assert!(!duplicate_exact_identity(&[], |_, _| panic!(
        "empty catalog comparison"
    )));
    assert!(!duplicate_exact_identity(
        &[(0, 0, 42), (0, 1, 43)],
        |_, _| { panic!("cross-bucket comparison") }
    ));
}
