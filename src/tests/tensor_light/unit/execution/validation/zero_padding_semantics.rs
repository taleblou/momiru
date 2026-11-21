// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::degenerate_axis_and_single_input::{validate_operation, value};
use crate::tensor_light::execution::{Operation, PadMode, PadSpec, PlanError, StaticShape};
use crate::tensor_light::MAX_RANK;

fn spec(mode: PadMode) -> PadSpec {
    PadSpec {
        before: [0; MAX_RANK],
        after: [0; MAX_RANK],
        mode,
        constant_f32_bits: 0,
    }
}

fn pad(spec: PadSpec, input: StaticShape, output: StaticShape) -> Result<(), PlanError> {
    validate_operation(
        Operation::Pad(spec),
        vec![value(0, input), value(1, output)],
        &[0],
        1,
    )
}

#[test]
fn zero_padding_is_identity_even_when_the_mode_would_need_a_source_index() {
    for mode in [
        PadMode::Constant,
        PadMode::Replicate,
        PadMode::Reflect,
        PadMode::Circular,
    ] {
        for shape in [
            StaticShape::Rank0,
            StaticShape::Rank2([1, 0]),
            StaticShape::Rank1([1]),
        ] {
            assert_eq!(pad(spec(mode), shape, shape), Ok(()), "{mode:?} {shape:?}");
        }
        assert_eq!(
            pad(spec(mode), StaticShape::Rank1([0]), StaticShape::Rank1([1])),
            Err(PlanError::InvalidOperationSpec)
        );
    }
}

#[test]
fn nonzero_padding_on_empty_axis_requires_constant_mode() {
    for mode in [
        PadMode::Constant,
        PadMode::Replicate,
        PadMode::Reflect,
        PadMode::Circular,
    ] {
        let mut padding = spec(mode);
        padding.before[0] = 1;
        padding.after[0] = 2;
        let expected = if mode == PadMode::Constant {
            Ok(())
        } else {
            Err(PlanError::InvalidOperationSpec)
        };
        assert_eq!(
            pad(padding, StaticShape::Rank1([0]), StaticShape::Rank1([3])),
            expected,
            "{mode:?}"
        );
    }
}

#[test]
fn unused_axis_fields_and_overflow_fail_even_when_no_input_element_is_addressable() {
    let mut padding = spec(PadMode::Constant);
    padding.after[MAX_RANK - 1] = 1;
    assert_eq!(
        pad(padding, StaticShape::Rank0, StaticShape::Rank0),
        Err(PlanError::InvalidOperationSpec)
    );

    padding = spec(PadMode::Constant);
    padding.after[0] = 1;
    let shape = StaticShape::Rank2([usize::MAX, 0]);
    assert_eq!(
        pad(padding, shape, shape),
        Err(PlanError::InvalidOperationSpec)
    );
}
