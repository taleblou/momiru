// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::semantic::validate_writable_strided_nonoverlap;
use crate::tensor_light::execution::{LayoutKind, PlanError, StaticShape, TensorLayout};
use crate::tensor_light::{Layout, MAX_RANK};

fn strided<const R: usize>(active: [isize; R]) -> TensorLayout {
    let mut strides = [0; MAX_RANK];
    strides[..R].copy_from_slice(&active);
    TensorLayout {
        kind: LayoutKind::Strided,
        strides,
        block: [0; MAX_RANK],
        alignment_bytes: 4,
    }
}

#[test]
fn writable_ir_admission_matches_an_independent_small_address_oracle() {
    let shape = StaticShape::Rank2([3, 2]);
    for strides in [[3, 4], [-3, 4], [2, 4], [0, 4], [-2, -4]] {
        let mut offsets = Vec::new();
        let mut distinct = true;
        for row in 0..3 {
            for col in 0..2 {
                let offset = row * strides[0] + col * strides[1];
                distinct &= !offsets.contains(&offset);
                offsets.push(offset);
            }
        }
        let expected = if distinct {
            Ok(())
        } else {
            Err(PlanError::InvalidStorageAccess)
        };
        assert_eq!(
            validate_writable_strided_nonoverlap(&strided(strides), &shape),
            expected,
            "{strides:?}"
        );
    }
}

#[test]
fn four_axis_collision_cannot_authorize_a_write_even_when_analyzer_is_inconclusive() {
    let dims = [2, 2, 2, 17];
    let strides = [1, 3, 7, -11];
    let layout = Layout::new(dims, strides, 176, 188).unwrap();
    assert_eq!(layout.checked_offset([0, 0, 0, 0]), Ok(176));
    assert_eq!(layout.checked_offset([1, 1, 1, 1]), Ok(176));
    assert_eq!(
        validate_writable_strided_nonoverlap(&strided(strides), &StaticShape::Rank4(dims)),
        Err(PlanError::InvalidStorageAccess)
    );
}

#[test]
fn empty_and_scalar_mappings_are_unique_but_nonempty_cardinality_must_fit() {
    let layout = strided([isize::MIN, 0, isize::MAX]);
    assert_eq!(
        validate_writable_strided_nonoverlap(&layout, &StaticShape::Rank3([usize::MAX, 0, 2])),
        Ok(())
    );
    assert_eq!(
        validate_writable_strided_nonoverlap(&layout, &StaticShape::Rank3([usize::MAX, 1, 2])),
        Err(PlanError::TensorByteCountOverflow)
    );
    assert_eq!(
        validate_writable_strided_nonoverlap(&strided([]), &StaticShape::Rank0),
        Ok(())
    );
}
