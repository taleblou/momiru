// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use momiru::tensor_light::{
    InternalOverlap, SignedSliceSpec, TensorError, TensorView, TensorViewMut,
};

#[test]
fn signed_borrow_preserves_gaps_and_rebases_offsets() -> Result<(), TensorError> {
    let storage = [-1.0, 10.0, 99.0, 20.0, 98.0, 30.0, -2.0];
    let view = TensorView::from_offset_strided([3], [2], 1, &storage)?;
    let signed = view.signed_view()?;
    assert_eq!(signed.origin_offset(), 0);
    assert_eq!(signed.layout().capacity(), 5);
    assert_eq!(signed.layout().reachable_bounds(), Some((0, 4)));
    let flipped = signed.flip(0)?;
    for (logical, physical) in [5, 3, 1].into_iter().enumerate() {
        assert!(core::ptr::eq(
            flipped.try_get_ref([logical])?,
            &storage[physical]
        ));
    }
    assert_eq!(flipped.as_slice(), Err(TensorError::NonContiguous));
    Ok(())
}

#[test]
fn borrowed_broadcast_keeps_shared_addresses_after_signed_slicing() -> Result<(), TensorError> {
    let storage = [1.0, 2.0, 3.0];
    let view = TensorView::new_strided([1, 3], [3, 1], &storage)?;
    let expanded = view.broadcast_to([2, 3])?;
    let signed = expanded
        .signed_view()?
        .slice_axis(1, SignedSliceSpec::reverse())?;
    assert_eq!(signed.strides(), &[0, -1]);
    assert_eq!(signed.internal_overlap(), InternalOverlap::Definite);
    assert!(core::ptr::eq(
        signed.try_get_ref([0, 0])?,
        signed.try_get_ref([1, 0])?
    ));
    assert!(core::ptr::eq(signed.try_get_ref([1, 0])?, &storage[2]));
    Ok(())
}

#[test]
fn mutable_signed_reborrow_writes_the_mapped_addresses_only() -> Result<(), TensorError> {
    let mut storage = [-1.0, 10.0, 99.0, 20.0, 98.0, 30.0, -2.0];
    {
        let mut view = TensorViewMut::from_offset_strided([3], [2], 1, &mut storage)?;
        let mut signed = view.signed_view_mut()?.flip_mut(0)?;
        assert_eq!(signed.layout().capacity(), 5);
        assert_eq!(signed.strides(), &[-2]);
        signed.set([0], 31.0)?;
        signed.set([2], 11.0)?;
        assert_eq!(signed.set([3], 0.0), Err(TensorError::IndexOutOfBounds));
    }
    assert_eq!(storage, [-1.0, 11.0, 99.0, 20.0, 98.0, 31.0, -2.0]);
    Ok(())
}

#[test]
fn empty_one_past_borrow_has_no_reachable_addresses() -> Result<(), TensorError> {
    let mut storage = [7.0];
    let mut empty = TensorViewMut::from_offset_strided(
        [usize::MAX, 0],
        [isize::MAX as usize, 0],
        1,
        &mut storage,
    )?;
    let mut signed = empty.signed_view_mut()?.flip_mut(0)?;
    assert_eq!(signed.dims(), &[usize::MAX, 0]);
    assert_eq!(signed.layout().capacity(), 0);
    assert_eq!(signed.layout().reachable_bounds(), None);
    assert_eq!(signed.as_mut_slice()?, &mut []);
    assert_eq!(signed.get([0, 0]), Err(TensorError::IndexOutOfBounds));
    assert_eq!(storage, [7.0]);
    Ok(())
}
