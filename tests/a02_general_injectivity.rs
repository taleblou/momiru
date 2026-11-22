// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use momiru::tensor::{Layout, TensorError, TensorViewMut};

#[test]
fn interleaved_signed_mutable_view_updates_only_its_elements() -> Result<(), TensorError> {
    let mut storage = [0; 11];
    let mut dims = [1; 9];
    dims[1] = 3;
    dims[8] = 2;
    let mut strides = [0; 9];
    strides[1] = 3;
    strides[8] = -4;

    {
        let mut view = TensorViewMut::new(dims, strides, 4, &mut storage)?;
        let mut index = [0; 9];
        for (row, offsets) in [[4, 0], [7, 3], [10, 6]].into_iter().enumerate() {
            index[1] = row;
            for (column, offset) in offsets.into_iter().enumerate() {
                index[8] = column;
                assert_eq!(view.layout().checked_offset(index)?, offset);
                *view.get_mut(index)? = offset + 1;
            }
        }
    }

    assert_eq!(storage, [1, 0, 0, 4, 5, 0, 7, 8, 0, 0, 11]);
    Ok(())
}

#[test]
fn mutable_admission_matches_enumerated_addresses_for_two_axes() {
    const ORIGIN: isize = 30;
    let mut storage = [0_u8; 61];

    for rows in 0..=4 {
        for columns in 0..=4 {
            for row_stride in -5..=5 {
                for column_stride in -5..=5 {
                    let mut occupied = [false; 61];
                    let mut unique = true;
                    for row in 0..rows {
                        for column in 0..columns {
                            let address = ORIGIN
                                + row as isize * row_stride
                                + column as isize * column_stride;
                            unique &= !core::mem::replace(&mut occupied[address as usize], true);
                        }
                    }

                    let dims = [rows, columns];
                    let strides = [row_stride, column_stride];
                    let result = TensorViewMut::new(dims, strides, ORIGIN, &mut storage);
                    if unique {
                        assert!(result.is_ok(), "{dims:?} with {strides:?}: {result:?}");
                    } else {
                        assert_eq!(
                            result.unwrap_err(),
                            TensorError::OverlappingMutableView,
                            "{dims:?} with {strides:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn opposite_large_strides_distinguish_neighbors_from_collisions() -> Result<(), TensorError> {
    let max = isize::MAX;
    let interleaved = Layout::new([2, 2], [max - 1, -max], max, usize::MAX)?;
    assert_eq!(interleaved.reachable_bounds(), Some((0, usize::MAX - 2)));
    assert!(interleaved.is_proven_unique());

    let overlapping = Layout::new([2, 2], [max, -max], max, usize::MAX)?;
    assert_eq!(
        overlapping.checked_offset([0, 0]),
        overlapping.checked_offset([1, 1])
    );
    assert!(!overlapping.is_proven_unique());
    Ok(())
}
