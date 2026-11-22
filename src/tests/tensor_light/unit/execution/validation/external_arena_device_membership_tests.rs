// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::external_arena_device_is_declared;

#[test]
fn host_external_arena_does_not_require_a_gpu_topology() {
    assert!(external_arena_device_is_declared(None, &[]));
    assert!(external_arena_device_is_declared(None, &[1, 7]));
    assert!(!external_arena_device_is_declared(Some(0), &[]));
}

#[test]
fn gpu_external_arena_requires_exact_frozen_memory_membership() {
    // Sparse device identifiers are compared by value, not by list position.
    // These descriptors exercise a validation predicate without a GPU launch.
    let frozen_ordinals = [1, 7, u16::MAX];
    for ordinal in [1, 7, u16::MAX] {
        assert!(external_arena_device_is_declared(
            Some(ordinal),
            &frozen_ordinals
        ));
    }
    for ordinal in [0, 2, 6, 8, u16::MAX - 1] {
        assert!(!external_arena_device_is_declared(
            Some(ordinal),
            &frozen_ordinals
        ));
    }
}
