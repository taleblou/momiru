// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use std::collections::BTreeSet;

use momiru::tensor_light::{InternalOverlap, Layout, Shape, SignedTensorViewMut, TensorError};

#[test]
fn common_stride_divisor_proves_collision_in_large_signed_layouts() -> Result<(), TensorError> {
    let dims = [2, 2, 2, 17, 1];
    for scale in [1isize, 2, 3, 5] {
        let strides = [scale, 2 * scale, 4 * scale, -7 * scale, 1];
        let origin = 112 * scale;
        let capacity = (119 * scale + 1) as usize;
        let mut storage = vec![0.0; capacity];
        let layout = Layout::new(dims, strides, origin, capacity)?;
        let mut addresses = BTreeSet::new();
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..17 {
                        let expected = ((112 + a + 2 * b + 4 * c - 7 * d) * scale) as usize;
                        let index = [a as usize, b as usize, c as usize, d as usize, 0];
                        assert_eq!(layout.checked_offset(index)?, expected);
                        addresses.insert(expected);
                    }
                }
            }
        }
        assert_eq!(addresses.len(), 120);
        assert_eq!(layout.numel(), 136);
        assert_eq!(layout.span_len(), capacity);
        assert_eq!(layout.internal_overlap(), InternalOverlap::Definite);
        assert!(matches!(
            SignedTensorViewMut::new(dims, strides, origin, &mut storage),
            Err(TensorError::OverlappingMutableView)
        ));

        let unsigned = Shape::strided(dims, strides.map(isize::unsigned_abs))?;
        assert_eq!(unsigned.internal_overlap(), InternalOverlap::Definite);
    }
    Ok(())
}

#[test]
fn exactly_enough_spaced_offsets_can_still_be_unique() -> Result<(), TensorError> {
    let dims = [2, 2, 2, 17];
    let strides = [2, 4, -8, 16];
    let mut storage = [0.0; 271];
    let mut view = SignedTensorViewMut::new(dims, strides, 8, &mut storage)?;
    assert_eq!(view.layout().numel(), 136);
    assert_eq!(view.layout().span_len(), 271);
    assert_eq!(view.layout().internal_overlap(), InternalOverlap::None);
    view.set([0, 0, 1, 0], 1.0)?;
    view.set([1, 1, 0, 16], 2.0)?;
    assert_eq!(storage[0], 1.0);
    assert_eq!(storage[270], 2.0);
    Ok(())
}
