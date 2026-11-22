// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Generic host tensors with runtime rank and signed-stride borrowed views.
//!
//! Owned tensors store initialized elements in row-major order. Views borrow
//! the original slice, and their offsets and strides are measured in elements.
//! An empty shape has no reachable addresses. Rank zero is a single element.
//!
//! ```
//! use momiru::tensor::Tensor;
//!
//! let mut tensor = Tensor::from_vec([2, 3], vec![0_i64, 1, 2, 3, 4, 5])?;
//! tensor[[1, 2]] = 42;
//! let reversed = tensor.view().flip(1)?;
//! assert_eq!(*reversed.get([1, 0])?, 42);
//! assert_eq!(reversed.layout().checked_offset([1, 0])?, 5);
//! # Ok::<(), momiru::tensor::TensorError>(())
//! ```

use core::fmt;
use core::num::NonZeroIsize;
use core::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorError {
    RankMismatch,
    ShapeMismatch,
    ShapeOverflow,
    LengthMismatch,
    IndexOutOfBounds,
    AxisOutOfBounds,
    InvalidPermutation,
    NonContiguous,
    OverlappingMutableView,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RankMismatch => "coordinate or stride count differs from tensor rank",
            Self::ShapeMismatch => "tensor shapes are incompatible",
            Self::ShapeOverflow => "tensor shape or stride exceeds representable bounds",
            Self::LengthMismatch => "storage length differs from tensor element count",
            Self::IndexOutOfBounds => "tensor index or storage offset is out of bounds",
            Self::AxisOutOfBounds => "tensor axis is out of bounds",
            Self::InvalidPermutation => "tensor axes do not form a permutation",
            Self::NonContiguous => "tensor layout is not forward row-major contiguous",
            Self::OverlappingMutableView => "tensor layout does not establish unique addresses",
        })
    }
}

impl std::error::Error for TensorError {}

fn element_count(dims: &[usize]) -> Result<usize, TensorError> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or(TensorError::ShapeOverflow)
    })
}

/// Validated runtime-rank address mapping over a backing slice of a given length.
///
/// Construction checks the entire reachable offset interval. A view that uses
/// the layout also checks the supplied slice length and owns its Rust borrow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout {
    dims: Vec<usize>,
    strides: Vec<isize>,
    origin: isize,
    len: usize,
    capacity: usize,
    bounds: Option<(usize, usize)>,
}

impl Layout {
    /// Create a dense row-major layout over exactly the required element count.
    pub fn contiguous(dims: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        let dims = dims.as_ref();
        let len = element_count(dims)?;
        let mut strides = vec![0; dims.len()];
        if len != 0 {
            let mut stride = 1usize;
            for (&dim, slot) in dims.iter().zip(&mut strides).rev() {
                *slot = match isize::try_from(stride) {
                    Ok(stride) => stride,
                    Err(_) if dim == 1 => 0, // This axis has no displacement.
                    Err(_) => return Err(TensorError::ShapeOverflow),
                };
                stride *= dim; // Every suffix product is bounded by the checked total.
            }
        }
        Ok(Self {
            dims: dims.to_vec(),
            strides,
            origin: 0,
            len,
            capacity: len,
            bounds: len.checked_sub(1).map(|last| (0, last)),
        })
    }

    /// Validate an origin and signed element strides against a backing length.
    /// An empty layout may have an origin one past the end of the backing slice.
    pub fn new(
        dims: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        origin: isize,
        capacity: usize,
    ) -> Result<Self, TensorError> {
        let dims = dims.as_ref();
        let strides = strides.as_ref();
        if dims.len() != strides.len() {
            return Err(TensorError::RankMismatch);
        }
        let len = element_count(dims)?;
        let start = usize::try_from(origin).map_err(|_| TensorError::IndexOutOfBounds)?;
        let bounds = if len == 0 {
            if start > capacity {
                return Err(TensorError::IndexOutOfBounds);
            }
            None
        } else {
            let mut low = origin as i128;
            let mut high = low;
            for (&dim, &stride) in dims.iter().zip(strides) {
                let displacement = ((dim - 1) as i128)
                    .checked_mul(stride as i128)
                    .ok_or(TensorError::ShapeOverflow)?;
                let bound = if displacement < 0 {
                    &mut low
                } else {
                    &mut high
                };
                *bound = bound
                    .checked_add(displacement)
                    .ok_or(TensorError::ShapeOverflow)?;
            }
            let low = usize::try_from(low).map_err(|_| TensorError::IndexOutOfBounds)?;
            let high = usize::try_from(high).map_err(|_| TensorError::IndexOutOfBounds)?;
            if high >= capacity {
                return Err(TensorError::IndexOutOfBounds);
            }
            Some((low, high))
        };
        Ok(Self {
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            origin,
            len,
            capacity,
            bounds,
        })
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn origin_offset(&self) -> isize {
        self.origin
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inclusive minimum and maximum backing offsets, if an element exists.
    pub fn reachable_bounds(&self) -> Option<(usize, usize)> {
        self.bounds
    }

    pub fn checked_offset(&self, index: impl AsRef<[usize]>) -> Result<usize, TensorError> {
        let index = index.as_ref();
        if index.len() != self.ndim() {
            return Err(TensorError::RankMismatch);
        }
        if self.is_empty() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let mut offset = self.origin as i128;
        for ((&coordinate, &dim), &stride) in index.iter().zip(&self.dims).zip(&self.strides) {
            if coordinate >= dim {
                return Err(TensorError::IndexOutOfBounds);
            }
            // Each term and each partial sum lie within the validated
            // displacement and offset bounds.
            offset += (coordinate as i128) * (stride as i128);
        }
        usize::try_from(offset).map_err(|_| TensorError::IndexOutOfBounds)
    }

    pub fn is_c_contiguous(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for (&dim, &stride) in self.dims.iter().zip(&self.strides).rev() {
            if dim > 1 && usize::try_from(stride) != Ok(expected) {
                return false;
            }
            expected *= dim;
        }
        true
    }

    /// Prove that distinct coordinates address distinct elements.
    ///
    /// The check is exact for up to two axes with extent greater than one.
    /// For more axes, each active axis must step beyond the interval occupied
    /// by all smaller stride magnitudes. This sufficient check can be
    /// inconclusive for interleaved layouts.
    pub fn is_proven_unique(&self) -> bool {
        if self.len <= 1 {
            return true;
        }
        let mut axes: Vec<_> = self
            .dims
            .iter()
            .zip(&self.strides)
            .filter(|&(&dim, _)| dim > 1)
            .map(|(&dim, &stride)| (stride.unsigned_abs(), dim))
            .collect();
        if let &[(stride_a, dim_a), (stride_b, dim_b)] = axes.as_slice() {
            if stride_a == 0 || stride_b == 0 {
                return false;
            }
            let (mut gcd, mut remainder) = (stride_a, stride_b);
            while remainder != 0 {
                (gcd, remainder) = (remainder, gcd % remainder);
            }
            // Any collision uses coordinate differences that are nonzero
            // multiples of (stride_b / gcd, stride_a / gcd) in magnitude.
            return dim_a <= stride_b / gcd || dim_b <= stride_a / gcd;
        }
        axes.sort_unstable();
        let mut span = 1u128;
        for (stride, dim) in axes {
            let stride = stride as u128;
            if stride < span {
                return false;
            }
            // The checked interval bounds the total displacement by usize::MAX.
            span += ((dim - 1) as u128) * stride;
        }
        true
    }

    fn contiguous_range(&self) -> Result<core::ops::Range<usize>, TensorError> {
        if !self.is_c_contiguous() {
            return Err(TensorError::NonContiguous);
        }
        let start = usize::try_from(self.origin).map_err(|_| TensorError::IndexOutOfBounds)?;
        let end = start
            .checked_add(self.len)
            .filter(|&end| end <= self.capacity)
            .ok_or(TensorError::IndexOutOfBounds)?;
        Ok(start..end)
    }

    fn permuted(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.ndim() {
            return Err(TensorError::InvalidPermutation);
        }
        let mut seen = vec![false; self.ndim()];
        let mut result = self.clone();
        for (dst, &src) in axes.iter().enumerate() {
            if src >= self.ndim() || seen[src] {
                return Err(TensorError::InvalidPermutation);
            }
            seen[src] = true;
            result.dims[dst] = self.dims[src];
            result.strides[dst] = self.strides[src];
        }
        Ok(result)
    }

    fn flipped(&self, axis: usize) -> Result<Self, TensorError> {
        let extent = *self.dims.get(axis).ok_or(TensorError::AxisOutOfBounds)?;
        let mut result = self.clone();
        if !self.is_empty() && extent > 1 {
            result.origin = isize::try_from(
                (self.origin as i128) + ((extent - 1) as i128) * (self.strides[axis] as i128),
            )
            .map_err(|_| TensorError::ShapeOverflow)?;
            result.strides[axis] = self.strides[axis]
                .checked_neg()
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Ok(result)
    }

    fn narrowed(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        let extent = *self.dims.get(axis).ok_or(TensorError::AxisOutOfBounds)?;
        if start > extent || length > extent - start {
            return Err(TensorError::IndexOutOfBounds);
        }
        let mut dims = self.dims.clone();
        dims[axis] = length;
        let origin = if length == 0 || self.is_empty() {
            self.origin
        } else {
            isize::try_from((self.origin as i128) + (start as i128) * (self.strides[axis] as i128))
                .map_err(|_| TensorError::ShapeOverflow)?
        };
        Self::new(dims, &self.strides, origin, self.capacity)
    }

    fn sliced_axis(
        &self,
        axis: usize,
        start: usize,
        length: usize,
        step: NonZeroIsize,
    ) -> Result<Self, TensorError> {
        let extent = *self.dims.get(axis).ok_or(TensorError::AxisOutOfBounds)?;
        if start > extent || (length != 0 && start == extent) {
            return Err(TensorError::IndexOutOfBounds);
        }
        let step = step.get() as i128;
        if length != 0 {
            let last = ((length - 1) as i128)
                .checked_mul(step)
                .and_then(|delta| (start as i128).checked_add(delta))
                .ok_or(TensorError::ShapeOverflow)?;
            if !usize::try_from(last).is_ok_and(|coordinate| coordinate < extent) {
                return Err(TensorError::IndexOutOfBounds);
            }
        }

        let mut dims = self.dims.clone();
        dims[axis] = length;
        let mut strides = self.strides.clone();
        let source_stride = strides[axis] as i128;
        let origin = if length == 0 || self.is_empty() {
            self.origin
        } else {
            let origin = (start as i128)
                .checked_mul(source_stride)
                .and_then(|delta| (self.origin as i128).checked_add(delta))
                .ok_or(TensorError::ShapeOverflow)?;
            isize::try_from(origin).map_err(|_| TensorError::ShapeOverflow)?
        };
        strides[axis] = if length <= 1 || self.is_empty() {
            0
        } else {
            let stride = source_stride
                .checked_mul(step)
                .ok_or(TensorError::ShapeOverflow)?;
            isize::try_from(stride).map_err(|_| TensorError::ShapeOverflow)?
        };
        Self::new(dims, strides, origin, self.capacity)
    }

    fn reshaped(&self, dims: &[usize]) -> Result<Self, TensorError> {
        if !self.is_c_contiguous() {
            return Err(TensorError::NonContiguous);
        }
        let mut result = Self::contiguous(dims)?;
        if self.len != result.len {
            return Err(TensorError::ShapeMismatch);
        }
        // Equal cardinality and forward order preserve the same reachable interval.
        result.origin = self.origin;
        result.capacity = self.capacity;
        result.bounds = self.bounds;
        Ok(result)
    }
}

/// Owned, initialized row-major host tensor with an arbitrary runtime rank.
#[derive(Clone, Debug)]
pub struct Tensor<T> {
    layout: Layout,
    storage: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn from_vec(dims: impl AsRef<[usize]>, storage: Vec<T>) -> Result<Self, TensorError> {
        let layout = Layout::contiguous(dims)?;
        if storage.len() != layout.len() {
            return Err(TensorError::LengthMismatch);
        }
        Ok(Self { layout, storage })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.dims()
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.storage
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage
    }

    pub fn get(&self, index: impl AsRef<[usize]>) -> Result<&T, TensorError> {
        Ok(&self.storage[self.layout.checked_offset(index)?])
    }

    pub fn get_mut(&mut self, index: impl AsRef<[usize]>) -> Result<&mut T, TensorError> {
        Ok(&mut self.storage[self.layout.checked_offset(index)?])
    }

    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            layout: self.layout.clone(),
            storage: &self.storage,
        }
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            layout: self.layout.clone(),
            storage: &mut self.storage,
        }
    }

    /// Change only the dimensions, preserving row-major order and allocation.
    pub fn reshape(self, dims: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.reshaped(dims.as_ref())?,
            storage: self.storage,
        })
    }
}

impl<T, I: AsRef<[usize]>> Index<I> for Tensor<T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect("invalid General tensor index")
    }
}

impl<T, I: AsRef<[usize]>> IndexMut<I> for Tensor<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect("invalid General tensor index")
    }
}

/// Shared host view. Negative strides can reverse axes and zero strides can
/// broadcast elements. Logical indices are allowed to share an address.
#[derive(Debug)]
pub struct TensorView<'a, T> {
    layout: Layout,
    storage: &'a [T],
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(
        dims: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        origin: isize,
        storage: &'a [T],
    ) -> Result<Self, TensorError> {
        Ok(Self {
            layout: Layout::new(dims, strides, origin, storage.len())?,
            storage,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn get(&self, index: impl AsRef<[usize]>) -> Result<&'a T, TensorError> {
        Ok(&self.storage[self.layout.checked_offset(index)?])
    }

    /// Expose the logical order as a slice if the layout is forward contiguous.
    pub fn as_slice(&self) -> Result<&'a [T], TensorError> {
        Ok(&self.storage[self.layout.contiguous_range()?])
    }

    pub fn permute(&self, axes: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.permuted(axes.as_ref())?,
            storage: self.storage,
        })
    }

    pub fn flip(&self, axis: usize) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.flipped(axis)?,
            storage: self.storage,
        })
    }

    /// Select a checked half-open range of coordinates along one axis.
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.narrowed(axis, start, length)?,
            storage: self.storage,
        })
    }

    /// Borrow `length` coordinates starting at `start` and advancing by `step`.
    /// All selected coordinates must lie within the axis. An empty selection
    /// permits `start` at the end of the axis and preserves the backing origin.
    pub fn slice_axis(
        &self,
        axis: usize,
        start: usize,
        length: usize,
        step: NonZeroIsize,
    ) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.sliced_axis(axis, start, length, step)?,
            storage: self.storage,
        })
    }

    /// Reshape in logical row-major order without moving any elements.
    pub fn reshape_view(&self, dims: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        Ok(Self {
            layout: self.layout.reshaped(dims.as_ref())?,
            storage: self.storage,
        })
    }

    /// Right-align the source axes. A singleton may expand to any extent,
    /// including zero, and expanded or leading axes use stride zero.
    pub fn broadcast_to(&self, dims: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        let dims = dims.as_ref();
        let padding = dims
            .len()
            .checked_sub(self.layout.ndim())
            .ok_or(TensorError::ShapeMismatch)?;
        let mut strides = vec![0; dims.len()];
        for (axis, &dim) in dims[padding..].iter().enumerate() {
            let source_dim = self.layout.dims[axis];
            if dim == source_dim {
                strides[padding + axis] = self.layout.strides[axis];
            } else if source_dim != 1 {
                return Err(TensorError::ShapeMismatch);
            }
        }
        Self::new(dims, strides, self.layout.origin, self.storage)
    }
}

/// Exclusive host view over a layout whose addresses have proven uniqueness.
/// Consuming transforms preserve the original exclusive borrow.
///
/// ```compile_fail
/// use momiru::tensor::Tensor;
///
/// let mut tensor = Tensor::from_vec([2], vec![1, 2]).unwrap();
/// let mut reversed = tensor.view_mut().flip(0).unwrap();
/// tensor[[0]] = 3;
/// *reversed.get_mut([0]).unwrap() = 4;
/// ```
#[derive(Debug)]
pub struct TensorViewMut<'a, T> {
    layout: Layout,
    storage: &'a mut [T],
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn new(
        dims: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        origin: isize,
        storage: &'a mut [T],
    ) -> Result<Self, TensorError> {
        let layout = Layout::new(dims, strides, origin, storage.len())?;
        if !layout.is_proven_unique() {
            return Err(TensorError::OverlappingMutableView);
        }
        Ok(Self { layout, storage })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn get(&self, index: impl AsRef<[usize]>) -> Result<&T, TensorError> {
        Ok(&self.storage[self.layout.checked_offset(index)?])
    }

    pub fn get_mut(&mut self, index: impl AsRef<[usize]>) -> Result<&mut T, TensorError> {
        Ok(&mut self.storage[self.layout.checked_offset(index)?])
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [T], TensorError> {
        Ok(&mut self.storage[self.layout.contiguous_range()?])
    }

    pub fn permute(mut self, axes: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        self.layout = self.layout.permuted(axes.as_ref())?;
        Ok(self)
    }

    pub fn flip(mut self, axis: usize) -> Result<Self, TensorError> {
        self.layout = self.layout.flipped(axis)?;
        Ok(self)
    }

    pub fn narrow(mut self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        self.layout = self.layout.narrowed(axis, start, length)?;
        Ok(self)
    }

    /// Select a stepped subset of the existing view, retaining its exclusive borrow.
    /// Coordinates follow the same rules as [`TensorView::slice_axis`].
    pub fn slice_axis(
        mut self,
        axis: usize,
        start: usize,
        length: usize,
        step: NonZeroIsize,
    ) -> Result<Self, TensorError> {
        self.layout = self.layout.sliced_axis(axis, start, length, step)?;
        Ok(self)
    }

    pub fn reshape_view(mut self, dims: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        self.layout = self.layout.reshaped(dims.as_ref())?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_owned_indexing_and_reshape_reuse_storage() -> Result<(), TensorError> {
        struct Value(i32);

        let mut tensor = Tensor::from_vec([2, 2], vec![Value(1), Value(2), Value(3), Value(4)])?;
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.len(), 4);
        tensor[[1, 0]] = Value(30);
        assert_eq!(tensor.get([1, 0])?.0, 30);
        let addr = tensor.as_slice().as_ptr();
        let reshaped = tensor.reshape([4, 1])?;
        assert_eq!(reshaped.shape(), &[4, 1]);
        assert_eq!(reshaped.as_slice().as_ptr(), addr);
        assert_eq!(reshaped[[2, 0]].0, 30);
        Ok(())
    }

    #[test]
    fn runtime_rank_scalar_and_empty_shapes_have_distinct_domains() -> Result<(), TensorError> {
        let scalar = Tensor::from_vec([], vec!["scalar"])?;
        assert_eq!(scalar[[]], "scalar");
        assert_eq!(scalar.layout().reachable_bounds(), Some((0, 0)));

        let mut dims = [1; 16];
        dims[10] = 2;
        let tensor = Tensor::from_vec(dims, vec![3, 5])?;
        let mut index = [0; 16];
        index[10] = 1;
        assert_eq!(tensor[index], 5);
        assert_eq!(*tensor.view().flip(10)?.get(index)?, 3);

        let empty = TensorView::new([usize::MAX, usize::MAX, 0], [isize::MIN, 1, 0], 1, &[7])?;
        assert!(empty.layout().is_empty());
        assert_eq!(empty.layout().reachable_bounds(), None);
        assert_eq!(empty.get([0, 0, 0]), Err(TensorError::IndexOutOfBounds));
        assert_eq!(empty.flip(0)?.reshape_view([0, 2])?.as_slice()?, &[]);
        Ok(())
    }

    #[test]
    fn signed_permutation_narrow_and_broadcast_match_original_addresses() -> Result<(), TensorError>
    {
        let storage = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let source = TensorView::new([2, 3], [6, -2], 4, &storage)?;
        let narrowed = source.permute([1, 0])?.flip(0)?.narrow(0, 1, 2)?;
        let result = narrowed.broadcast_to([4, 2, 2])?;
        assert_eq!(result.layout().reachable_bounds(), Some((2, 10)));
        assert_eq!(result.as_slice(), Err(TensorError::NonContiguous));
        assert!(matches!(
            result.reshape_view([16]),
            Err(TensorError::NonContiguous)
        ));
        for batch in 0..4 {
            for col in 0..2 {
                for row in 0..2 {
                    let expected = 2 + 2 * col + 6 * row;
                    let index = [batch, col, row];
                    assert_eq!(result.layout().checked_offset(index)?, expected);
                    assert!(core::ptr::eq(result.get(index)?, &storage[expected]));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mutable_views_reject_aliases_and_reverse_only_the_selected_elements(
    ) -> Result<(), TensorError> {
        let mut storage = [0, 1, 2, 3, 4, 5, 6];
        for strides in [[0, 1], [1, 1], [-1, 1]] {
            assert!(matches!(
                TensorViewMut::new([2, 2], strides, 1, &mut storage),
                Err(TensorError::OverlappingMutableView)
            ));
        }
        let mut reversed = TensorViewMut::new([3], [2], 1, &mut storage)?
            .flip(0)?
            .narrow(0, 1, 2)?;
        assert_eq!(reversed.as_mut_slice(), Err(TensorError::NonContiguous));
        *reversed.get_mut([0])? = 30;
        *reversed.get_mut([1])? = 10;
        assert_eq!(storage, [0, 10, 2, 30, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn shape_bounds_permutation_and_broadcast_errors_are_checked() -> Result<(), TensorError> {
        assert_eq!(
            Layout::contiguous([usize::MAX, 2]),
            Err(TensorError::ShapeOverflow)
        );
        assert_eq!(Layout::new([2], [], 0, 2), Err(TensorError::RankMismatch));
        assert_eq!(
            Layout::new([2], [-1], 0, 2),
            Err(TensorError::IndexOutOfBounds)
        );
        assert_eq!(
            Layout::new([2], [1], 1, 2),
            Err(TensorError::IndexOutOfBounds)
        );
        assert_eq!(
            Layout::new([0], [0], -1, 2),
            Err(TensorError::IndexOutOfBounds)
        );
        assert!(matches!(
            Tensor::from_vec([2], vec![0]),
            Err(TensorError::LengthMismatch)
        ));

        let source = Tensor::from_vec([1, 2], vec![3, 4])?;
        assert_eq!(source.get([0]), Err(TensorError::RankMismatch));
        assert!(matches!(
            source.view().permute([0, 0]),
            Err(TensorError::InvalidPermutation)
        ));
        assert!(matches!(
            source.view().narrow(1, usize::MAX, 1),
            Err(TensorError::IndexOutOfBounds)
        ));
        assert!(matches!(
            source.view().reshape_view([3]),
            Err(TensorError::ShapeMismatch)
        ));
        let empty = source.view().broadcast_to([0, 2])?;
        assert!(empty.layout().is_empty());
        assert!(matches!(
            empty.broadcast_to([1, 2]),
            Err(TensorError::ShapeMismatch)
        ));
        Ok(())
    }

    #[test]
    fn forward_subview_reshape_preserves_nonzero_origin() -> Result<(), TensorError> {
        let mut data = [0, 1, 2, 3, 4, 5];
        let mut subview = TensorViewMut::new([2, 2], [2, 1], 2, &mut data)?.reshape_view([4])?;
        assert_eq!(subview.layout().origin_offset(), 2);
        assert_eq!(subview.as_mut_slice()?, &mut [2, 3, 4, 5]);
        subview.as_mut_slice()?.copy_from_slice(&[7, 8, 9, 10]);
        assert_eq!(data, [0, 1, 7, 8, 9, 10]);
        Ok(())
    }
}
