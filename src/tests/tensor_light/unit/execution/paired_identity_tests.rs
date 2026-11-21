// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use core::cell::Cell;
use core::hash::{Hash, Hasher};

use super::{stable_hash_pair, StableIdentityHasher};

struct PrimitiveWrites<'a>(&'a Cell<usize>);

impl Hash for PrimitiveWrites<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.set(self.0.get() + 1);
        state.write(&[0x01, 0x80, 0xff]);
        state.write(&[]);
        state.write_u8(0x80);
        state.write_u16(0x1234);
        state.write_u32(0x1234_5678);
        state.write_u64(0x1020_3040_5060_7080);
        state.write_u128(u128::MAX - 8);
        state.write_usize(usize::MAX);
        state.write_i8(i8::MIN);
        state.write_i16(-145);
        state.write_i32(i32::MIN);
        state.write_i64(-1_234_567);
        state.write_i128(i128::MIN);
        state.write_isize(-42);
    }
}

fn two_pass_reference<T: Hash + ?Sized>(value: &T, domain: u64) -> (u64, u64) {
    let mut hi = StableIdentityHasher::new(0xcbf2_9ce4_8422_2325 ^ domain);
    let mut lo = StableIdentityHasher::new(0x8422_2325_cbf2_9ce4 ^ domain.rotate_left(23));
    value.hash(&mut hi);
    value.hash(&mut lo);
    (hi.finish(), lo.finish())
}

#[test]
fn forwards_every_primitive_writer_to_both_lanes_in_one_traversal() {
    let calls = Cell::new(0);
    let value = PrimitiveWrites(&calls);
    for domain in [0, 0x0123_4567_89ab_cdef, u64::MAX] {
        calls.set(0);
        let paired = stable_hash_pair(&value, domain);
        assert_eq!(calls.get(), 1);
        assert_eq!(paired, two_pass_reference(&value, domain));
        assert_eq!(calls.get(), 3);
    }
}

#[test]
fn unsized_and_nested_inputs_keep_the_two_pass_identity() {
    let empty: &[u32] = &[];
    let nonempty: &[u32] = &[u32::MAX, 0, 17, 0];
    for domain in [5, 23] {
        assert_eq!(
            stable_hash_pair(empty, domain),
            two_pass_reference(empty, domain)
        );
        assert_eq!(
            stable_hash_pair(nonempty, domain),
            two_pass_reference(nonempty, domain)
        );
        let nested = (nonempty, None::<u64>, [Some(-1_i32), Some(3_i32)]);
        assert_eq!(
            stable_hash_pair(&nested, domain),
            two_pass_reference(&nested, domain)
        );
    }
    // Check this fixture responds to a domain change, without claiming
    // collision resistance or a cross-toolchain wire identity.
    assert_ne!(
        stable_hash_pair(nonempty, 5),
        stable_hash_pair(nonempty, 23)
    );
}
