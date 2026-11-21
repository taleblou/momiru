# Momiru

Momiru is an MIT-licensed Rust tensor library with independent General and Light editions.

## Tensor General

`momiru::tensor::Tensor<T>` owns generic host elements in row-major order and takes its rank at runtime. Its borrowed views support checked indexing, permutation, reversal, slicing, and forward-contiguous zero-copy reshape. Shared views can broadcast with zero strides. Mutable views require a checked address interval and unique element offsets. The General API covers host storage and views.

```rust
use momiru::tensor::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::from_vec([2, 3], vec![1_i64, 2, 3, 4, 5, 6])?;
    let reversed = tensor.view().flip(1)?;
    assert_eq!(*reversed.get([1, 0])?, 6);
    assert_eq!(reversed.layout().checked_offset([1, 0])?, 5);
    Ok(())
}
```

See the [General layout guide](docs/a02/momiru-general-layouts.md) for writable signed-stride views.

## Tensor Light

The `momiru::tensor_light` module exposes owned FP32 `Tensor<R>` values and borrowed views for const ranks 0 through 8. Its signed-stride views support checked indexing, slicing, axis permutation, broadcasting, and zero-copy flattening under documented layout constraints. Read-only views can share physical elements. Mutable views require a layout with proven unique addresses.

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

Use the Rust 1.99.1 toolchain pinned in [rust-toolchain.toml](rust-toolchain.toml). The Cargo package has no third-party Rust dependencies. Run the repository checks from its root directory.

```sh
cargo fmt --all -- --check
cargo check --locked --all-targets --all-features
cargo check --locked --benches
cargo test --locked
cargo test --locked --features aot
cargo clippy --locked --all-targets --all-features
```

## Authors and license

Momiru and its separate Mojo sibling, [Momijo](https://github.com/taleblou/momijo), were co-founded and authored by **Morteza Taleblou** and **Mitra Daneshmand**. Both authors are credited in Momiru's [MIT license](LICENSE).
