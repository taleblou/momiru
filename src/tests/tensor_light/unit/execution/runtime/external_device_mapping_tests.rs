// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::external_device_matches_frozen_topology;

#[test]
fn frozen_membership_is_required_even_when_the_arena_names_the_device() {
    let topology = [1, 4, u16::MAX];
    for ordinal in topology {
        assert!(external_device_matches_frozen_topology(
            ordinal, &topology, None
        ));
        assert!(external_device_matches_frozen_topology(
            ordinal,
            &topology,
            Some(ordinal)
        ));
    }
    assert!(!external_device_matches_frozen_topology(
        0,
        &topology,
        Some(0)
    ));
    assert!(!external_device_matches_frozen_topology(3, &topology, None));
    assert!(!external_device_matches_frozen_topology(4, &[], Some(4)));
}

#[test]
fn affinity_to_a_frozen_arena_cannot_be_satisfied_by_another_admitted_device() {
    let topology = [2, 7];
    assert!(external_device_matches_frozen_topology(
        2,
        &topology,
        Some(2)
    ));
    assert!(!external_device_matches_frozen_topology(
        7,
        &topology,
        Some(2)
    ));
    assert!(!external_device_matches_frozen_topology(
        2,
        &topology,
        Some(7)
    ));
    assert!(!external_device_matches_frozen_topology(
        2,
        &topology,
        Some(9)
    ));
}
