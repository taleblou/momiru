// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::SplitMix64;
use crate::tensor_light::{Tensor, TensorError};

#[test]
fn bernoulli_distinguishes_adjacent_words_below_the_old_24_bit_resolution() {
    let probability = f32::from_bits((127 - 40) << 23); // 2^-40
    let threshold = 1u64 << 24;
    let source = Tensor::<1>::zeros_aligned([1], 256).unwrap();
    // Inverting the SplitMix64 output permutation gives seeds whose first
    // words lie immediately on opposite sides of this exact threshold.
    for (seed, word, expected) in [
        (0x74e5_f319_db80_529d, threshold - 1, 1.0),
        (0x0cd1_4bd4_2303_1708, threshold, 0.0),
    ] {
        assert_eq!(SplitMix64::new(seed).next_u64(), word);
        assert_eq!(word >> 40, 0);
        assert_eq!(
            Tensor::<1>::bernoulli([1], probability, seed)
                .unwrap()
                .as_slice(),
            &[expected]
        );
        let like = source.bernoulli_like(probability, seed).unwrap();
        assert_eq!(like.as_slice(), &[expected]);
        assert_eq!(like.alignment(), 256);
    }
}

#[test]
fn bernoulli_threshold_handles_one_grid_point_and_truncates_below_the_grid() {
    let grid_spacing = f32::from_bits((127 - 64) << 23);
    let below_grid = f32::from_bits((127 - 65) << 23);
    let below_one = f32::from_bits(1.0_f32.to_bits() - 1);
    assert_eq!(SplitMix64::bernoulli_threshold(grid_spacing), 1);
    assert_eq!(SplitMix64::bernoulli_threshold(below_grid), 0);
    assert_eq!(
        SplitMix64::bernoulli_threshold(below_one),
        u64::MAX - ((1 << 40) - 1)
    );
}

#[test]
fn bernoulli_endpoints_and_invalid_probabilities_are_checked() {
    for (probability, bits) in [(0.0, 0), (-0.0, 0), (1.0, 1.0_f32.to_bits())] {
        let mask = Tensor::<1>::bernoulli([4], probability, u64::MAX).unwrap();
        assert!(mask.as_slice().iter().all(|value| value.to_bits() == bits));
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        assert!(matches!(
            Tensor::<1>::bernoulli([0], invalid, 0),
            Err(TensorError::InvalidProbability)
        ));
    }
}
