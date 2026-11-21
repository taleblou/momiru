# A02 host layout behavior in Momiru Light

`momiru::tensor_light::Layout<R>` describes an FP32 host mapping for rank 0 through 8. Given an allocation-relative origin, the element offset of a valid index is the origin plus the sum of each coordinate times its signed stride. Construction checks the logical element count and reachable physical bounds. `Layout::checked_offset` checks each coordinate and returns an element offset relative to the original slice. Equal logical values can appear at different physical offsets, so value equality alone says nothing about storage aliasing.

- Rank zero addresses one scalar. A zero extent gives an empty index domain, with no reachable bounds and no valid element index. An empty origin may be one past the end of the backing slice.
- Negative strides reverse physical traversal along an axis. A zero stride on a length-one axis contributes no motion. A zero stride on an axis of length greater than one repeats addresses in a nonempty view.
- `SignedTensorView::broadcast_to` right-aligns dimensions and expands singleton axes with stride zero. A singleton can broadcast to extent zero. An extent of zero cannot broadcast to one.
- `SignedTensorViewMut` requires the overlap classifier to establish uniqueness. Both definite and undecided overlap classifications reject mutable view construction. The view holds an exclusive slice borrow, and consuming transforms preserve that borrow.
- The overlap classifier can prove a collision when the logical element count exceeds the number of offsets permitted by the active strides' common divisor. The physical span still includes all gaps. Having enough candidate offsets leaves this check inconclusive.
- `reshape_view` requires forward C-contiguous order and equal logical element counts. `flatten_view` merges a selected group of adjacent axes. A nonempty view requires an affine C-order progression through that group, which can have a negative or zero stride. When a zero extent elsewhere makes the whole view empty, a non-affine group can use stride zero. The merged extent still has to fit in `usize`. A group containing a zero extent merges to zero without multiplying its other extents.

Call `TensorView::signed_view` or `TensorViewMut::signed_view_mut` to apply signed operations to an existing nonnegative-stride view without copying. The conversion keeps the same shape, strides, and backing span. Signed offsets start at zero relative to that span, even if the original view was created at an offset into a larger slice. The mutable result reborrows the original view exclusively.

## Focused validation

Run the Light integration tests from the repository root.

```sh
cargo test --locked --test a02_light_layout --test a02_borrowed_signed_views --test a02_stride_spacing
cargo test --locked --features aot --test a02_light_layout --test a02_borrowed_signed_views --test a02_stride_spacing
```

The suite checks rank and overflow boundaries, signed slicing, empty flattening, address mapping, and writable alias rejection. Its bounded enumeration covers 441 rank-two layouts. Scientific trials and canonical protocol conformance are `NOT_RUN`. Cross-engine pairing is `INELIGIBLE` pending the canonical corpus, evidence protocols, and vectors and independent validation of those inputs.
