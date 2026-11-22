// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::randperm_length_is_exact_f32;
use crate::tensor_light::{Tensor, TensorError};

#[test]
fn length_boundary_is_one_past_the_largest_consecutive_exact_integer() {
    let max_integer = 1usize << f32::MANTISSA_DIGITS;
    for len in [0, 1, max_integer, max_integer + 1] {
        assert!(randperm_length_is_exact_f32(len));
    }
    for len in [max_integer + 2, usize::MAX] {
        assert!(!randperm_length_is_exact_f32(len));
        // Invalid lengths must fail before trying to create a tensor.
        assert!(matches!(
            Tensor::<1>::randperm(len, 0),
            Err(TensorError::InvalidRange)
        ));
    }
}

#[test]
fn seeded_permutations_are_reproducible_and_include_each_value_once() {
    for len in [0, 1, 2, 17, 257] {
        let first = Tensor::<1>::randperm(len, 42).unwrap();
        let again = Tensor::<1>::randperm(len, 42).unwrap();
        assert_eq!(first.as_slice(), again.as_slice());
        let mut actual = first.as_slice().to_vec();
        actual.sort_unstable_by(f32::total_cmp);
        let expected: Vec<_> = (0..len).map(|index| index as f32).collect();
        assert_eq!(actual, expected);
    }
}
