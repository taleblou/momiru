// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::arena_guarantees_slot_alignment;

#[test]
fn aligned_slot_offset_does_not_compensate_for_weak_arena_base_alignment() {
    for offset in [0, 128, 256] {
        assert!(!arena_guarantees_slot_alignment(64, 128, offset));
        assert!(arena_guarantees_slot_alignment(128, 128, offset));
        assert!(arena_guarantees_slot_alignment(256, 128, offset));
    }
}

#[test]
fn both_guarantees_must_be_powers_of_two_and_the_offset_must_be_aligned() {
    for (arena, slot, offset) in [
        (0, 16, 0),
        (64, 0, 0),
        (96, 16, 32),
        (128, 24, 48),
        (128, 32, 33),
    ] {
        assert!(!arena_guarantees_slot_alignment(arena, slot, offset));
    }
    assert!(arena_guarantees_slot_alignment(16, 1, usize::MAX));
    assert!(!arena_guarantees_slot_alignment(64, 32, usize::MAX));
}
