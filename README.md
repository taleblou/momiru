# Momiru

Momiru is an MIT-licensed Rust tensor library with independent General and Light editions. The Light edition source in this checkout implements fixed-rank FP32 host tensors for ranks 0 through 8. The crate declares the separate `momiru::tensor` General edition, whose source is missing from this checkout. Full validation also requires the unit-test and benchmark sources referenced by the repository.

## Tensor Light

The `momiru::tensor_light` module exposes owned `Tensor<R>` values and borrowed views. Its signed-stride views support checked indexing, slicing, axis permutation, broadcasting, and zero-copy flattening under documented layout constraints. Read-only views can share physical elements. Mutable views require a layout with proven unique addresses.

```rust
use momiru::tensor_light::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::<2>::try_from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let reversed = tensor.flip_view(1)?;
    assert_eq!(reversed.get([1, 0])?, 6.0);
    assert_eq!(reversed.layout().checked_offset([1, 0])?, 5);
    Ok(())
}
```

See the [A02 host layout guide](docs/a02/momiru-light-layouts.md) for layout and aliasing rules and the focused Light validation procedure.

## Setup and tests

Use the Rust 1.99.1 toolchain pinned in [rust-toolchain.toml](rust-toolchain.toml). The Cargo package has no third-party Rust dependencies. Run the repository gates from the root of a source-complete checkout.

```sh
cargo fmt --all -- --check
cargo check --locked --all-targets --all-features
cargo test --locked
cargo test --locked --features aot
cargo clippy --locked --all-targets --all-features
```

The focused Light procedure can exercise the new integration tests with the files present here. The repository gates remain blocked pending restoration of the General module, linked unit tests, and benchmark sources.

## Authors and license

Momiru and its separate Mojo sibling, [Momijo](https://github.com/taleblou/momijo), were co-founded and authored by **Morteza Taleblou** and **Mitra Daneshmand**. Both authors are credited in Momiru's [MIT license](LICENSE).
