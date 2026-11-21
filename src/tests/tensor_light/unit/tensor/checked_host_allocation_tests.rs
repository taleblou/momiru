// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::storage::checked_host_allocation;
use crate::tensor_light::{Tensor, TensorError, UninitTensor};

#[test]
fn allocator_null_returns_a_resource_error_and_valid_pointer_is_unchanged() {
    assert_eq!(
        checked_host_allocation::<f32>(core::ptr::null_mut()),
        Err(TensorError::AllocationFailed)
    );
    let mut element = 1.0_f32;
    let pointer = &raw mut element;
    let checked = checked_host_allocation::<f32>(pointer.cast()).unwrap();
    assert_eq!(checked.as_ptr(), pointer);
}

#[test]
fn constructors_check_product_and_byte_capacity_before_allocating() {
    assert!(matches!(
        Tensor::<2>::zeros([usize::MAX, 2]),
        Err(TensorError::ShapeOverflow)
    ));
    let too_many = isize::MAX as usize / size_of::<f32>() + 1;
    assert!(matches!(
        Tensor::<1>::zeros([too_many]),
        Err(TensorError::AllocationOverflow)
    ));
    assert!(matches!(
        UninitTensor::<1>::new([too_many]),
        Err(TensorError::AllocationOverflow)
    ));
    for alignment in [0, 3] {
        assert!(matches!(
            Tensor::<1>::zeros_aligned([0], alignment),
            Err(TensorError::InvalidAlignment)
        ));
        assert!(matches!(
            UninitTensor::<1>::new_aligned([1], alignment),
            Err(TensorError::InvalidAlignment)
        ));
    }
}

#[test]
fn zero_elements_and_initialized_allocations_preserve_requested_alignment() {
    let empty = Tensor::<3>::zeros_aligned([usize::MAX, 0, usize::MAX], 256).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.alignment(), 256);
    assert_eq!(empty.as_ptr().addr() % 256, 0);

    let full = Tensor::<2>::full_aligned([2, 3], -0.0, 256).unwrap();
    assert_eq!(full.alignment(), 256);
    assert_eq!(full.as_ptr().addr() % 256, 0);
    assert!(full
        .as_slice()
        .iter()
        .all(|value| value.to_bits() == (-0.0_f32).to_bits()));
}
