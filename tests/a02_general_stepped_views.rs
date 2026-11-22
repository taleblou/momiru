// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use core::num::NonZeroIsize;
use momiru::tensor::{TensorError, TensorView, TensorViewMut};
use momiru::tensor_light::{IndexResult, PredicateMask};

#[test]
fn stepped_view_composes_with_negative_strides_and_writes_back() -> Result<(), TensorError> {
    let mut storage = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let step = NonZeroIsize::new(-2).unwrap();
    {
        let mut view = TensorViewMut::new([4], [-2], 8, &mut storage)?.slice_axis(0, 3, 2, step)?;
        assert_eq!(view.layout().strides(), &[4]);
        assert_eq!(view.layout().checked_offset([0])?, 2);
        assert_eq!(view.layout().checked_offset([1])?, 6);
        *view.get_mut([0])? = -2;
        *view.get_mut([1])? = -6;
    }
    assert_eq!(storage, [0, 1, -2, 3, 4, 5, -6, 7, 8]);
    Ok(())
}

#[test]
fn stepped_broadcast_retains_addresses_and_rejects_out_of_range_coordinates(
) -> Result<(), TensorError> {
    let view = TensorView::new([5], [0], 1, &[11, 42])?;
    let step = NonZeroIsize::new(-2).unwrap();
    let reversed = view.slice_axis(0, 4, 3, step)?;
    assert_eq!(reversed.layout().strides(), &[0]);
    for index in 0..3 {
        assert_eq!(reversed.layout().checked_offset([index])?, 1);
        assert_eq!(*reversed.get([index])?, 42);
    }
    assert_eq!(
        view.slice_axis(0, 4, 4, step).unwrap_err(),
        TensorError::IndexOutOfBounds
    );
    assert_eq!(
        view.slice_axis(0, 5, 1, step).unwrap_err(),
        TensorError::IndexOutOfBounds
    );
    assert_eq!(
        view.slice_axis(1, 0, 0, step).unwrap_err(),
        TensorError::AxisOutOfBounds
    );
    let empty = view.slice_axis(0, 5, 0, step)?;
    assert_eq!(empty.layout().origin_offset(), 1);
    assert_eq!(empty.layout().reachable_bounds(), None);
    Ok(())
}

#[test]
fn extreme_step_checks_coordinates_without_inventing_physical_motion() -> Result<(), TensorError> {
    let step = NonZeroIsize::new(isize::MIN).unwrap();
    let view = TensorView::new([2], [2], 0, &[11, 0, 42])?;
    let singleton = view.slice_axis(0, 1, 1, step)?;
    assert_eq!(singleton.layout().strides(), &[0]);
    assert_eq!(*singleton.get([0])?, 42);
    assert_eq!(
        view.slice_axis(0, 1, 2, step).unwrap_err(),
        TensorError::IndexOutOfBounds
    );

    let empty = TensorView::<u8>::new([0, usize::MAX], [0, isize::MIN], 0, &[])?;
    let sliced = empty.slice_axis(1, usize::MAX - 1, 2, step)?;
    assert_eq!(sliced.layout().dims(), &[0, 2]);
    assert_eq!(sliced.layout().reachable_bounds(), None);
    Ok(())
}

#[test]
fn light_predicate_and_index_results_distinguish_empty_from_scalar() {
    assert!(PredicateMask::try_from_vec([2, 0], vec![])
        .unwrap()
        .is_empty());
    assert!(!PredicateMask::try_from_vec([], vec![0]).unwrap().is_empty());
    assert!(IndexResult::try_from_vec([2, 0], vec![])
        .unwrap()
        .is_empty());
    assert!(!IndexResult::try_from_vec([], vec![0]).unwrap().is_empty());
}
