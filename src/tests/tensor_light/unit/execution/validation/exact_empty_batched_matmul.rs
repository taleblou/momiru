// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::degenerate_axis_and_single_input::{validate_operation, value};
use crate::tensor_light::execution::{Operation, PlanError, StaticShape};

fn matmul(
    lhs: StaticShape,
    rhs: StaticShape,
    out: StaticShape,
    transposes: [bool; 2],
) -> Result<(), PlanError> {
    validate_operation(
        Operation::MatMul {
            transpose_lhs: transposes[0],
            transpose_rhs: transposes[1],
        },
        vec![value(0, lhs), value(1, rhs), value(2, out)],
        &[0, 1],
        2,
    )
}

#[test]
fn batch_broadcast_preserves_zero_instead_of_promoting_it_to_one() {
    let lhs = StaticShape::Rank4([2, 1, 2, 3]);
    let rhs = StaticShape::Rank3([0, 3, 4]);
    assert_eq!(
        matmul(lhs, rhs, StaticShape::Rank4([2, 0, 2, 4]), [false; 2]),
        Ok(())
    );
    assert_eq!(
        matmul(lhs, rhs, StaticShape::Rank4([2, 1, 2, 4]), [false; 2]),
        Err(PlanError::InvalidOperationSpec)
    );
    assert_eq!(
        matmul(
            StaticShape::Rank4([2, 2, 2, 3]),
            rhs,
            StaticShape::Rank4([2, 0, 2, 4]),
            [false; 2]
        ),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn transposes_do_not_remove_matrix_shape_checks_when_batch_is_empty() {
    let lhs = StaticShape::Rank3([0, 3, 2]);
    let rhs = StaticShape::Rank2([4, 3]);
    let output = StaticShape::Rank3([0, 2, 4]);
    assert_eq!(matmul(lhs, rhs, output, [true; 2]), Ok(()));
    assert_eq!(
        matmul(lhs, rhs, StaticShape::Rank3([0, 4, 2]), [true; 2]),
        Err(PlanError::InvalidOperationSpec)
    );
    assert_eq!(
        matmul(lhs, StaticShape::Rank2([4, 5]), output, [true; 2]),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn zero_contraction_can_still_have_a_nonempty_output() {
    assert_eq!(
        matmul(
            StaticShape::Rank3([2, 3, 0]),
            StaticShape::Rank3([1, 0, 4]),
            StaticShape::Rank3([2, 3, 4]),
            [false; 2]
        ),
        Ok(())
    );
    assert_eq!(
        matmul(
            StaticShape::Rank1([0]),
            StaticShape::Rank1([0]),
            StaticShape::Rank0,
            [false; 2]
        ),
        Ok(())
    );
}
