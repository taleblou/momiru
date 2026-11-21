// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Measures Light's zero-copy Vec adoption and separately allocated aligned
//! copy. Source preparation, verification, and destruction are untimed.

use momiru::tensor_light::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const SHAPE: [usize; 2] = [128, 257];
const LEN: usize = SHAPE[0] * SHAPE[1];
const ALIGNMENT: usize = 256;

fn check(output: &Tensor<2>, reference: &[f32]) {
    assert_eq!(output.dims(), &SHAPE);
    assert_eq!(output.len(), reference.len());
    assert!(output
        .as_slice()
        .iter()
        .zip(reference)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
}

fn adopt(reference: &[f32]) -> Result<u128, TensorError> {
    let input = reference.to_vec();
    let pointer = input.as_ptr();
    let start = Instant::now();
    let output = Tensor::<2>::try_from_vec(black_box(SHAPE), black_box(input))?;
    black_box(&output);
    let time = start.elapsed().as_nanos();
    assert_eq!(
        output.as_ptr(),
        pointer,
        "Vec adoption must keep its buffer"
    );
    check(&output, reference);
    Ok(time)
}

fn copy_aligned(reference: &[f32]) -> Result<u128, TensorError> {
    let start = Instant::now();
    let output =
        Tensor::<2>::from_slice_aligned(black_box(SHAPE), black_box(reference), ALIGNMENT)?;
    black_box(&output);
    let time = start.elapsed().as_nanos();
    assert_ne!(output.as_ptr(), reference.as_ptr());
    assert_eq!(output.alignment(), ALIGNMENT);
    assert_eq!(output.as_ptr().addr() % ALIGNMENT, 0);
    check(&output, reference);
    Ok(time)
}

fn measure(
    operation: &str,
    scope: &str,
    mut run: impl FnMut() -> Result<u128, TensorError>,
) -> Result<(), TensorError> {
    run()?;
    let mut samples = [0; 9];
    for sample in &mut samples {
        *sample = run()?;
    }
    samples.sort_unstable();
    println!(
        "implementation=light_host operation={operation} scope={scope} dtype=f32 elements={LEN} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let mut reference: Vec<_> = (0..LEN).map(|i| (i % 257) as f32 / 16.0).collect();
    reference[..4].copy_from_slice(&[-0.0, 0.0, f32::from_bits(0x7fc0_0123), f32::INFINITY]);
    measure("adopt_vec", "constructor_only", || adopt(&reference))?;
    measure("copy_aligned", "allocation_and_copy", || {
        copy_aligned(&reference)
    })
}
