// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use momiru::tensor_light::{
    InternalOverlap, Layout, Shape, SignedSliceSpec, SignedTensorView, SignedTensorViewMut, Tensor,
    TensorError, TensorView,
};

#[test]
fn flatten_empty_view_across_non_affine_axes() -> Result<(), TensorError> {
    let storage = [42.0];
    let signed = SignedTensorView::new([0, 2, 3], [0, -1, 2], 1, &storage)?;
    let flat = signed.flatten_view::<2>(1, 2)?;
    assert_eq!(flat.dims(), &[0, 6]);
    assert_eq!(flat.strides(), &[0, 0]);
    assert_eq!(flat.origin_offset(), 1);
    assert_eq!(flat.layout().reachable_bounds(), None);
    assert_eq!(flat.as_slice()?, &[]);
    assert_eq!(flat.get([0, 0]), Err(TensorError::IndexOutOfBounds));

    let unsigned = TensorView::new(Shape::strided([0, 2, 3], [0, 1, 2])?, &storage)?;
    assert_eq!(unsigned.flatten_view::<2>(1, 2)?.dims(), &[0, 6]);

    let mut storage = storage;
    let unique = SignedTensorViewMut::new([0, 2, 3], [0, -1, 2], 1, &mut storage)?;
    let mut flat = unique.flatten_view_mut::<2>(1, 2)?;
    assert_eq!(flat.dims(), &[0, 6]);
    assert_eq!(flat.as_mut_slice()?, &mut []);
    Ok(())
}

#[test]
fn flatten_checks_the_merged_extent_even_when_another_axis_is_empty() -> Result<(), TensorError> {
    for strides in [[0, 0, 0], [0, 1, 1]] {
        let view = SignedTensorView::new([0, usize::MAX, 2], strides, 0, &[])?;
        assert!(matches!(
            view.flatten_view::<2>(1, 2),
            Err(TensorError::ShapeOverflow)
        ));
        assert_eq!(view.flatten_view::<1>(0, 2)?.dims(), &[0]);
    }
    Ok(())
}

#[test]
fn signed_transforms_preserve_affine_offsets_and_backing_values() -> Result<(), TensorError> {
    let tensor = Tensor::<2>::try_from_vec([2, 3], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0])?;
    let signed = tensor.signed_view()?;
    let transposed = signed.transpose(0, 1)?;
    assert!(matches!(
        transposed.flatten_view::<1>(0, 1),
        Err(TensorError::NonContiguous)
    ));
    assert!(matches!(
        transposed.reshape_view([6]),
        Err(TensorError::NonContiguous)
    ));
    let flipped = signed.flip(0)?.flip(1)?.flatten_view::<1>(0, 1)?;
    let expanded = flipped.unsqueeze::<2>(0)?.broadcast_to([2, 6])?;
    for row in 0..2 {
        for col in 0..6 {
            let expected = 5 - col;
            assert_eq!(expanded.layout().checked_offset([row, col])?, expected);
            assert!(core::ptr::eq(
                expanded.try_get_ref([row, col])?,
                &tensor.as_slice()[expected]
            ));
        }
    }
    assert_eq!(expanded.internal_overlap(), InternalOverlap::Definite);
    assert_eq!(flipped.as_slice(), Err(TensorError::NonContiguous));
    Ok(())
}

#[test]
fn signed_slices_distinguish_omitted_and_explicit_negative_stop() -> Result<(), TensorError> {
    let view = SignedTensorView::new([3], [2], 1, &[0.0, 11.0, 0.0, 22.0, 0.0, 33.0])?;
    let reversed = view.flip(0)?;
    assert_eq!(reversed.get([0])?, 33.0);
    assert_eq!(reversed.get([2])?, 11.0);
    assert!(view
        .slice_axis(0, SignedSliceSpec::new(None, Some(-1), -1))?
        .is_empty());

    let singleton = view.slice_axis(0, SignedSliceSpec::new(None, None, isize::MIN))?;
    assert_eq!(singleton.dims(), &[1]);
    assert_eq!(singleton.get([0])?, 33.0);
    assert!(matches!(
        view.slice_axis(0, SignedSliceSpec::new(None, None, 0)),
        Err(TensorError::InvalidSliceStep)
    ));
    Ok(())
}

#[test]
fn signed_mutation_rejects_repeated_addresses_and_updates_only_mapped_elements(
) -> Result<(), TensorError> {
    let mut data = [0.0, 1.0, 2.0, 3.0, 4.0];
    for strides in [[0, 1], [1, 1], [-1, 1]] {
        assert!(matches!(
            SignedTensorViewMut::new([2, 2], strides, 1, &mut data),
            Err(TensorError::OverlappingMutableView)
        ));
    }
    let mut view = SignedTensorViewMut::new([1, 3], [0, -2], 4, &mut data)?
        .squeeze_axis_mut::<1>(0)?
        .slice_axis_mut(0, SignedSliceSpec::new(None, None, 2))?;
    view.set([0], 44.0)?;
    view.set([1], 11.0)?;
    assert_eq!(data, [11.0, 1.0, 2.0, 3.0, 44.0]);
    Ok(())
}

#[test]
fn small_signed_layouts_match_enumerated_addresses_and_aliases() -> Result<(), TensorError> {
    for rows in 1..=3 {
        for cols in 1..=3 {
            for row_stride in -3..=3 {
                for col_stride in -3..=3 {
                    let offsets: Vec<_> = (0..rows)
                        .flat_map(|r| (0..cols).map(move |c| 12 + r * row_stride + c * col_stride))
                        .collect();
                    let layout = Layout::new(
                        [rows as usize, cols as usize],
                        [row_stride, col_stride],
                        12,
                        25,
                    )?;
                    assert_eq!(
                        layout.reachable_bounds(),
                        Some((
                            *offsets.iter().min().unwrap() as usize,
                            *offsets.iter().max().unwrap() as usize
                        ))
                    );
                    for (linear, &offset) in offsets.iter().enumerate() {
                        let index = [linear / cols as usize, linear % cols as usize];
                        assert_eq!(layout.checked_offset(index)?, offset as usize);
                    }
                    let unique = offsets
                        .iter()
                        .enumerate()
                        .all(|(i, x)| !offsets[..i].contains(x));
                    let overlap = if unique {
                        InternalOverlap::None
                    } else {
                        InternalOverlap::Definite
                    };
                    assert_eq!(layout.internal_overlap(), overlap);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn writable_views_require_proven_uniqueness() -> Result<(), TensorError> {
    let mut data = [0.0; 188];
    let dims = [2, 2, 2, 17];
    let strides = [1, 3, 7, -11];
    let layout = Layout::new(dims, strides, 176, data.len())?;
    assert_ne!(layout.internal_overlap(), InternalOverlap::None);
    assert_eq!(
        layout.checked_offset([0, 0, 0, 0])?,
        layout.checked_offset([1, 1, 1, 1])?
    );
    assert!(matches!(
        SignedTensorViewMut::new(dims, strides, 176, &mut data),
        Err(TensorError::OverlappingMutableView)
    ));
    Ok(())
}

#[test]
fn scalar_max_rank_and_invalid_layout_boundaries() -> Result<(), TensorError> {
    let scalar = SignedTensorView::new([], [], 1, &[9.0, 8.0])?;
    assert_eq!(scalar.get([])?, 8.0);
    assert_eq!(scalar.numel(), 1);
    let max_rank = scalar.broadcast_to([1, 2, 1, 1, 1, 1, 1, 3])?;
    assert_eq!(max_rank.numel(), 6);
    assert_eq!(
        max_rank.layout().checked_offset([0, 1, 0, 0, 0, 0, 0, 2])?,
        1
    );
    assert_eq!(max_rank.layout().span_len(), 1);
    assert_eq!(
        Layout::new([], [], 1, 1),
        Err(TensorError::IndexOutOfBounds)
    );
    assert_eq!(
        Layout::new([2], [-1], 0, 2),
        Err(TensorError::IndexOutOfBounds)
    );
    assert_eq!(
        Layout::new([1], [0], -1, 1),
        Err(TensorError::IndexOutOfBounds)
    );
    assert_eq!(
        Layout::new([2], [1], 1, 2),
        Err(TensorError::IndexOutOfBounds)
    );
    assert_eq!(
        Layout::new([0], [0], 3, 2),
        Err(TensorError::IndexOutOfBounds)
    );
    assert_eq!(
        Layout::new([usize::MAX, 2], [0, 0], 0, 1),
        Err(TensorError::ShapeOverflow)
    );
    assert_eq!(
        Layout::new([usize::MAX], [0], 0, 1),
        Err(TensorError::AllocationOverflow)
    );
    Ok(())
}
