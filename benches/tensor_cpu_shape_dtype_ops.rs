// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Measures checked runtime-rank view composition and typed copying through
//! the General host API. The timed region includes view metadata creation
//! and indexed reads into preallocated output storage.

use momiru::tensor::{Tensor, TensorError, TensorView};
use std::{fmt::Debug, hint::black_box, time::Instant};

const SOURCE: [usize; 3] = [3, 5, 257];
const OUTPUT: [usize; 3] = [5, 2, 257];

fn transformed<T>(input: &Tensor<T>) -> Result<TensorView<'_, T>, TensorError> {
    input.view().permute([1, 0, 2])?.flip(2)?.narrow(1, 1, 2)
}

fn gather<T: Copy>(input: &Tensor<T>, output: &mut [T]) -> Result<(), TensorError> {
    let view = transformed(input)?;
    let [rows, cols, depth] = OUTPUT;
    for row in 0..rows {
        for col in 0..cols {
            for d in 0..depth {
                output[(row * cols + col) * depth + d] = *view.get([row, col, d])?;
            }
        }
    }
    Ok(())
}

fn verify<T: Debug + PartialEq>(input: &Tensor<T>, output: &[T]) -> Result<(), TensorError> {
    let view = transformed(input)?;
    assert_eq!(view.layout().dims(), &OUTPUT);
    let [rows, cols, depth] = OUTPUT;
    for row in 0..rows {
        for col in 0..cols {
            for d in 0..depth {
                let original = ((col + 1) * SOURCE[1] + row) * depth + (depth - 1 - d);
                let destination = (row * cols + col) * depth + d;
                assert_eq!(view.layout().checked_offset([row, col, d])?, original);
                assert_eq!(output[destination], input.as_slice()[original]);
            }
        }
    }
    Ok(())
}

fn measure<T: Copy + Debug + PartialEq>(
    dtype: &str,
    fill: impl Fn(usize) -> T,
) -> Result<(), TensorError> {
    let input = Tensor::from_vec(SOURCE, (0..SOURCE.iter().product()).map(&fill).collect())?;
    let mut output = Tensor::from_vec(OUTPUT, vec![fill(0); OUTPUT.iter().product()])?;
    gather(&input, output.as_mut_slice())?;
    verify(&input, output.as_slice())?;

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        gather(black_box(&input), black_box(output.as_mut_slice()))?;
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(&input, output.as_slice())?;
    samples.sort_unstable();
    println!(
        "implementation=general_host operation=permute_flip_narrow_gather dtype={dtype} elements={} iterations={} median_ns={}",
        output.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    measure("f32", |i| i as f32 / 16.0)?;
    measure("f64", |i| {
        (i % 127) as f64 + (i % 11) as f64 / (1u64 << 35) as f64
    })?;
    measure("u64", |i| (1 << 60) + i as u64)
}
