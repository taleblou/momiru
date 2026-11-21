// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::external_binding_covers_arena;

#[test]
fn foreign_root_cannot_cover_just_its_payload_or_the_end_of_its_first_slot() {
    let first_slot_offset = 128;
    let storage_bytes = 32;
    let first_slot_end = first_slot_offset + storage_bytes;
    let arena_bytes = 256;
    // The later slot belongs to the same immutable arena. A root buffer that
    // reaches the first slot (or is one byte short) is still insufficient.
    for available in [0, storage_bytes, first_slot_end, arena_bytes - 1] {
        assert!(!external_binding_covers_arena(available, arena_bytes));
    }
    for available in [arena_bytes, arena_bytes + 1] {
        assert!(external_binding_covers_arena(available, arena_bytes));
    }
}

#[test]
fn exact_zero_and_maximum_capacities_do_not_need_address_arithmetic() {
    assert!(external_binding_covers_arena(0, 0));
    assert!(!external_binding_covers_arena(0, 1));
    assert!(external_binding_covers_arena(usize::MAX, usize::MAX));
    assert!(!external_binding_covers_arena(usize::MAX - 1, usize::MAX));
}
