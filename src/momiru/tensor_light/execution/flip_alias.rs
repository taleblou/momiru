// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{AxisList, PlanError, StaticShape, TensorLayout};

/// Check the exact frozen strides and return the displacement in elements.
pub(super) fn flipped_alias_delta(
    shape: &StaticShape,
    axes: AxisList,
    source: &TensorLayout,
    output: &TensorLayout,
) -> Result<i128, PlanError> {
    let flipped_axes = axes
        .axes
        .get(..usize::from(axes.len))
        .ok_or(PlanError::InvalidKernelFamily)?;
    let mut delta = 0i128;
    for axis in 0..shape.rank() {
        let flipped = flipped_axes.contains(&(axis as u8));
        let source_stride = source.strides[axis];
        let expected_stride = if flipped {
            source_stride
                .checked_neg()
                .ok_or(PlanError::InvalidKernelFamily)?
        } else {
            source_stride
        };
        if output.strides[axis] != expected_stride {
            return Err(PlanError::InvalidKernelFamily);
        }
        if flipped {
            let extent = shape.dim(axis).ok_or(PlanError::InvalidKernelFamily)?;
            let extent = i128::try_from(extent.saturating_sub(1))
                .map_err(|_| PlanError::InvalidKernelFamily)?;
            let displacement = extent
                .checked_mul(source_stride as i128)
                .ok_or(PlanError::InvalidKernelFamily)?;
            delta = delta
                .checked_add(displacement)
                .ok_or(PlanError::InvalidKernelFamily)?;
        }
    }
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_light::execution::LayoutKind;
    use crate::tensor_light::MAX_RANK;

    fn strided(axis0: isize, axis1: isize) -> TensorLayout {
        let mut strides = [0; MAX_RANK];
        strides[0] = axis0;
        strides[1] = axis1;
        TensorLayout {
            kind: LayoutKind::Strided,
            strides,
            block: [0; MAX_RANK],
            alignment_bytes: 4,
        }
    }

    #[test]
    fn rejects_unrepresentable_flipped_stride() {
        let shape = StaticShape::Rank1([1]);
        let source = strided(isize::MIN, 0);
        let axes = AxisList {
            len: 1,
            ..AxisList::empty()
        };
        assert_eq!(source.validate_for(&shape), Ok(()));
        assert_eq!(
            flipped_alias_delta(&shape, axes, &source, &source),
            Err(PlanError::InvalidKernelFamily)
        );
    }

    #[test]
    fn reversing_negative_stride_retains_checked_displacement() {
        let shape = StaticShape::Rank2([3, 2]);
        let source = strided(-2, 1);
        let output = strided(2, 1);
        let axes = AxisList {
            len: 1,
            ..AxisList::empty()
        };
        assert_eq!(flipped_alias_delta(&shape, axes, &source, &output), Ok(-4));
        assert_eq!(
            flipped_alias_delta(&shape, axes, &source, &strided(2, 0)),
            Err(PlanError::InvalidKernelFamily)
        );
    }

    #[test]
    fn flipping_two_axes_preserves_every_address_in_a_gapped_or_broadcast_layout() {
        let shape = StaticShape::Rank3([3, 1, 4]);
        let mut axes = AxisList::empty();
        axes.len = 2;
        axes.axes[..2].copy_from_slice(&[2, 0]);
        assert_eq!(axes.validate_for_rank(shape.rank()), Ok(()));

        for inner_stride in [0isize, 2] {
            let mut source = strided(-5, isize::MIN);
            source.strides[2] = inner_stride;
            let mut output = strided(5, isize::MIN);
            output.strides[2] = -inner_stride;

            assert_eq!(source.validate_for(&shape), Ok(()));
            assert_eq!(output.validate_for(&shape), Ok(()));
            let delta = flipped_alias_delta(&shape, axes, &source, &output).unwrap();
            for row in 0..3i128 {
                for col in 0..4i128 {
                    let source_address = 10 - 5 * (2 - row) + (inner_stride as i128) * (3 - col);
                    let output_address = 10 + delta + 5 * row - (inner_stride as i128) * col;
                    assert_eq!(output_address, source_address);
                }
            }

            // A plausible transposed or copied stride descriptor can still
            // have valid metadata but must fail the exact alias check.
            output.strides[0] = 3;
            assert_eq!(output.validate_for(&shape), Ok(()));
            assert_eq!(
                flipped_alias_delta(&shape, axes, &source, &output),
                Err(PlanError::InvalidKernelFamily)
            );
        }
    }
}
