// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::degenerate_axis_and_single_input::{validate_operation, value};
use crate::tensor_light::execution::{
    NarrowSpec, Operation, PlanError, RepeatInterleaveSpec, StaticShape,
};

fn unary(operation: Operation, input: StaticShape, output: StaticShape) -> Result<(), PlanError> {
    validate_operation(operation, vec![value(0, input), value(1, output)], &[0], 1)
}

#[test]
fn reshape_and_expand_preserve_empty_domains_under_different_rules() {
    let oversized_empty = StaticShape::Rank3([usize::MAX, usize::MAX, 0]);
    assert_eq!(
        unary(Operation::Reshape, oversized_empty, StaticShape::Rank1([0])),
        Ok(())
    );
    assert_eq!(
        unary(Operation::Reshape, oversized_empty, StaticShape::Rank0),
        Err(PlanError::InvalidOperationSpec)
    );

    let singleton = StaticShape::Rank2([1, 3]);
    let empty = StaticShape::Rank2([0, 3]);
    assert_eq!(unary(Operation::Expand, singleton, empty), Ok(()));
    assert_eq!(
        unary(Operation::Expand, empty, singleton),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn narrow_accepts_an_empty_interval_at_the_end_and_rejects_beyond_end() {
    let input = StaticShape::Rank2([3, 2]);
    let output = StaticShape::Rank2([0, 2]);
    let at_end = NarrowSpec {
        axis: 0,
        start: 3,
        length: 0,
    };
    assert_eq!(unary(Operation::Narrow(at_end), input, output), Ok(()));
    for spec in [
        NarrowSpec { start: 4, ..at_end },
        NarrowSpec {
            length: 1,
            ..at_end
        },
    ] {
        assert_eq!(
            unary(Operation::Narrow(spec), input, output),
            Err(PlanError::InvalidOperationSpec)
        );
    }
}

#[test]
fn zero_insertion_and_repeat_interleave_check_distinct_output_rules() {
    for (input, factor, expected) in [(0, 7, 0), (1, 7, 1), (4, 1, 4), (3, 4, 9)] {
        assert_eq!(
            unary(
                Operation::ZeroInsert1d { factor },
                StaticShape::Rank1([input]),
                StaticShape::Rank1([expected])
            ),
            Ok(())
        );
    }
    assert_eq!(
        unary(
            Operation::ZeroInsert1d { factor: 4 },
            StaticShape::Rank1([3]),
            StaticShape::Rank1([12])
        ),
        Err(PlanError::InvalidOperationSpec)
    );
    for repeats in [0, 1, 3] {
        let repeat = Operation::RepeatInterleave(RepeatInterleaveSpec { axis: 0, repeats });
        assert_eq!(
            unary(
                repeat,
                StaticShape::Rank0,
                StaticShape::Rank1([repeats as usize])
            ),
            Ok(())
        );
    }
    assert_eq!(
        unary(
            Operation::RepeatInterleave(RepeatInterleaveSpec {
                axis: 0,
                repeats: 1
            }),
            StaticShape::Rank0,
            StaticShape::Rank0
        ),
        Err(PlanError::InvalidOperationSpec)
    );
}
