// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::SplitMix64;
use crate::tensor_light::Tensor;

#[test]
fn precomputed_threshold_matches_the_reference_and_keeps_rng_state() {
    for bound in [1, 2, 3, 4_294_967_297, (1u64 << 63) + 1, u64::MAX] {
        let threshold = ((1u128 << 64) % u128::from(bound)) as u64;
        assert_eq!(SplitMix64::rejection_threshold(bound), threshold);
        for seed in [0, 0x61c8_8646_80b5_83eb, u64::MAX] {
            let mut cached = SplitMix64::new(seed);
            let mut uncached = SplitMix64::new(seed);
            let mut oracle = SplitMix64::new(seed);
            for _ in 0..64 {
                let expected = loop {
                    let scaled = u128::from(oracle.next_u64()) * u128::from(bound);
                    let radix = 1u128 << 64;
                    if scaled % radix >= u128::from(threshold) {
                        break (scaled / radix) as u64;
                    }
                };
                assert_eq!(
                    cached.next_bounded_with_threshold(bound, threshold),
                    expected
                );
                assert_eq!(uncached.next_bounded(bound), expected);
                assert!(expected < bound);
            }
            assert_eq!(cached.next_u64(), oracle.next_u64());
        }
    }
}

#[test]
fn bounded_sampling_skips_a_known_rejected_word() {
    // The first SplitMix state is zero, so its output is zero and the
    // multiply-low word falls below the threshold of one for bound three.
    let seed = 0x61c8_8646_80b5_83eb;
    let mut reference = SplitMix64::new(seed);
    assert_eq!(reference.next_u64(), 0);
    let accepted = u128::from(reference.next_u64()) * 3;
    assert!(accepted as u64 >= 1);
    let mut tested = SplitMix64::new(seed);
    assert_eq!(
        tested.next_bounded_with_threshold(3, 1),
        (accepted >> 64) as u64
    );
    assert_eq!(tested.next_u64(), reference.next_u64());
}

#[test]
fn randint_and_randint_like_share_order_and_respect_half_open_bounds() {
    let source = Tensor::<2>::zeros_aligned([5, 9], 256).unwrap();
    let direct = Tensor::<2>::randint([5, 9], -7, 4, 123).unwrap();
    let like = source.randint_like(-7, 4, 123).unwrap();
    assert_eq!(direct.as_slice(), like.as_slice());
    assert_eq!(like.alignment(), 256);
    assert!(like
        .as_slice()
        .iter()
        .all(|value| value.fract() == 0.0 && (-7.0..4.0).contains(value)));
}
