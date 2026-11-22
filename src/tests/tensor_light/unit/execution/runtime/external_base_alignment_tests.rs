// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::external_base_alignment;

#[test]
fn every_frozen_alignment_can_be_the_strictest_requirement() {
    for (storage, slot, arena) in [(256, 64, 16), (16, 256, 64), (64, 16, 256), (256, 256, 256)] {
        assert_eq!(external_base_alignment(storage, slot, arena), 256);
    }
    assert_eq!(external_base_alignment(1, 1, 1), 1);
}

#[test]
fn a_slot_aligned_relative_to_the_arena_does_not_align_the_foreign_base() {
    let required = external_base_alignment(4, 32, 128);
    let foreign_base = 0x1020_usize;
    let slot_offset = 32;
    assert_eq!(slot_offset % 32, 0);
    assert_eq!((foreign_base + slot_offset) % 32, 0);
    assert_ne!(foreign_base % required, 0);
    assert_eq!(0x1080 % required, 0);
}
