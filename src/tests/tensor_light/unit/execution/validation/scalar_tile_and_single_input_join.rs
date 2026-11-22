// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::degenerate_axis_and_single_input::{validate_operation, value};
use crate::tensor_light::execution::{Operation, PlanError, StaticShape, TileSpec};
use crate::tensor_light::MAX_RANK;

#[test]
fn scalar_tile_is_rank_preserving_with_canonical_inactive_repeats() {
    let tile = TileSpec {
        rank: 0,
        repeats: [1; MAX_RANK],
    };
    let scalar = value(0, StaticShape::Rank0);
    assert_eq!(
        validate_operation(
            Operation::Tile(tile),
            vec![scalar, value(1, StaticShape::Rank0)],
            &[0],
            1
        ),
        Ok(())
    );
    assert_eq!(
        validate_operation(
            Operation::Tile(tile),
            vec![scalar, value(1, StaticShape::Rank1([1]))],
            &[0],
            1
        ),
        Err(PlanError::InvalidOperationSpec)
    );

    let mut bad_repeat = tile;
    bad_repeat.repeats[MAX_RANK - 1] = 2;
    assert_eq!(
        validate_operation(
            Operation::Tile(bad_repeat),
            vec![scalar, value(1, StaticShape::Rank0)],
            &[0],
            1
        ),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn single_input_joins_respect_scalar_and_max_rank_boundaries() {
    assert_eq!(
        validate_operation(
            Operation::Concat { axis: 0 },
            vec![value(0, StaticShape::Rank0), value(1, StaticShape::Rank0)],
            &[0],
            1
        ),
        Err(PlanError::InvalidOperationSpec)
    );
    let input = StaticShape::Rank7([2, 1, 1, 1, 1, 1, 0]);
    let rank8 = StaticShape::Rank8([2, 1, 1, 1, 1, 1, 0, 1]);
    assert_eq!(
        validate_operation(
            Operation::Stack { axis: 7 },
            vec![value(0, input), value(1, rank8)],
            &[0],
            1
        ),
        Ok(())
    );
    assert_eq!(
        validate_operation(
            Operation::Concat { axis: 7 },
            vec![value(0, rank8), value(1, rank8)],
            &[0],
            1
        ),
        Ok(())
    );
    assert_eq!(
        validate_operation(
            Operation::Stack { axis: 0 },
            vec![value(0, rank8), value(1, rank8)],
            &[0],
            1
        ),
        Err(PlanError::InvalidOperationSpec)
    );
}
