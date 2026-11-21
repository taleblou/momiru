// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::try_candidate_buffer;

#[test]
fn reserves_room_without_initializing_candidate_entries() {
    let mut entries = try_candidate_buffer::<String>(4).expect("small reservation");
    assert!(entries.is_empty());
    let capacity = entries.capacity();
    assert!(capacity >= 4);
    entries.extend(["first", "second", "third", "fourth"].map(String::from));
    assert_eq!(entries.capacity(), capacity);
    assert_eq!(entries, ["first", "second", "third", "fourth"]);
}

#[test]
fn unrepresentable_candidate_buffers_are_rejected_before_allocation() {
    let exceeds_object_bound = (isize::MAX as usize / size_of::<u64>()) + 1;
    assert!(try_candidate_buffer::<u64>(exceeds_object_bound).is_err());
    assert!(try_candidate_buffer::<u64>(usize::MAX).is_err());
}

#[test]
fn zero_capacity_and_zero_sized_metadata_need_no_initial_entries() {
    let empty = try_candidate_buffer::<u64>(0).expect("zero reservation");
    assert!(empty.is_empty());
    assert_eq!(empty.capacity(), 0);

    let zero_sized = try_candidate_buffer::<()>(usize::MAX).expect("zero-sized reservation");
    assert!(zero_sized.is_empty());
    assert_eq!(zero_sized.capacity(), usize::MAX);
}
