// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::degenerate_axis_and_single_input::{tensor, validate_operation, value};
use crate::tensor_light::execution::{
    AxisMask, AxisSelector, EmptyReductionIdentity, Operation, PlanError, ReductionKind,
    StaticShape, TensorRole, TensorSpec,
};
use crate::tensor_light::DType;

fn reduce(
    kind: ReductionKind,
    axis: u8,
    keep_dims: bool,
    input: TensorSpec,
    output: TensorSpec,
) -> Result<(), PlanError> {
    validate_operation(
        Operation::Reduction {
            kind,
            axes: AxisSelector::Explicit(AxisMask(1 << axis)),
            keep_dims,
        },
        vec![input, output],
        &[0],
        1,
    )
}

fn mask(id: u32, shape: StaticShape) -> TensorSpec {
    tensor(id, shape, DType::U8, TensorRole::Mask)
}

#[test]
fn evaluated_empty_slices_have_typed_boolean_and_count_identities() {
    let input = StaticShape::Rank2([3, 0]);
    let output = StaticShape::Rank1([3]);
    for (kind, expected) in [(ReductionKind::Any, 0), (ReductionKind::All, 1)] {
        assert_eq!(
            kind.empty_reduction_identity(),
            Some(EmptyReductionIdentity::MaskByte(expected))
        );
        assert_eq!(
            reduce(kind, 1, false, mask(0, input), mask(1, output)),
            Ok(())
        );
        assert_eq!(
            reduce(kind, 1, false, value(0, input), mask(1, output)),
            Err(PlanError::InvalidOperationSpec)
        );
    }

    for (kind, expected) in [
        (ReductionKind::AnyNonZero, 0),
        (ReductionKind::AllNonZero, 1),
        (ReductionKind::AnyPositive, 0),
        (ReductionKind::AllPositive, 1),
    ] {
        assert_eq!(
            kind.empty_reduction_identity(),
            Some(EmptyReductionIdentity::MaskByte(expected))
        );
        assert_eq!(
            reduce(kind, 1, false, value(0, input), mask(1, output)),
            Ok(())
        );
        assert_eq!(
            reduce(kind, 1, false, value(0, input), value(1, output)),
            Err(PlanError::InvalidOperationSpec)
        );
    }

    let count = ReductionKind::CountNonZero;
    assert_eq!(
        count.empty_reduction_identity(),
        Some(EmptyReductionIdentity::F32Bits(0))
    );
    assert_eq!(
        reduce(
            count,
            1,
            true,
            value(0, input),
            value(1, StaticShape::Rank2([3, 1]))
        ),
        Ok(())
    );
    assert_eq!(
        reduce(count, 1, false, value(0, input), mask(1, output)),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn a_zero_element_output_does_not_evaluate_any_empty_slices() {
    let kind = ReductionKind::Mean;
    assert_eq!(kind.empty_reduction_identity(), None);
    assert_eq!(
        reduce(
            kind,
            1,
            false,
            value(0, StaticShape::Rank2([0, 0])),
            value(1, StaticShape::Rank1([0]))
        ),
        Ok(())
    );
    assert_eq!(
        reduce(
            kind,
            1,
            false,
            value(0, StaticShape::Rank2([3, 0])),
            value(1, StaticShape::Rank1([3]))
        ),
        Err(PlanError::InvalidOperationSpec)
    );
}
