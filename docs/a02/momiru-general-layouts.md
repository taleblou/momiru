# A02 host layout behavior in Momiru General

`momiru::tensor::Layout` describes a runtime-rank host view. The element offset is the origin plus the sum of each coordinate times its signed stride. `Layout::new` checks the shape product, stride count, origin, and reachable interval against the backing length. `Layout::checked_offset` validates the coordinates and returns an offset relative to the original slice.

- Rank zero addresses one element. A zero extent makes the index domain empty and permits an origin at the end of the backing slice.
- Negative strides reverse traversal. On shared views, a zero stride along an axis of extent greater than one repeats addresses. Singleton axes contribute zero displacement regardless of their stride.
- `TensorViewMut::new` borrows its entire backing slice exclusively and requires unique offsets. The uniqueness check is exact for layouts with at most two axes of extent greater than one, including interleaved strides. With three or more such axes, it requires each successive stride magnitude to exceed the range covered by all smaller strides. Some injective layouts therefore return `TensorError::OverlappingMutableView`.
- `reshape_view` requires equal element counts and forward row-major contiguity. Permutation, reversal, and narrowing preserve the backing borrow.

Run the focused native tests from the repository root with `cargo test --locked --test a02_general_injectivity`. They check mutation of a rank-nine signed view against expected backing offsets and compare mutable view admission with independently enumerated addresses for small two-axis shapes.
