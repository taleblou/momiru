// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! FP64 row-major GEMM using General host tensors. The benchmark-local CPU
//! reference reuses an output allocation and checks it with an indexed dot oracle.

use momiru::tensor::{Tensor, TensorError};
use std::hint::black_box;
use std::time::Instant;

fn gemm(lhs: &[f64], rhs: &[f64], out: &mut [f64], inner: usize, cols: usize) {
    for (a, c) in lhs.chunks_exact(inner).zip(out.chunks_exact_mut(cols)) {
        c.fill(0.0);
        for (&left, b) in a.iter().zip(rhs.chunks_exact(cols)) {
            for (cell, &right) in c.iter_mut().zip(b) {
                *cell += left * right;
            }
        }
    }
}

fn check(
    lhs: &Tensor<f64>,
    rhs: &Tensor<f64>,
    output: &Tensor<f64>,
    [rows, inner, cols]: [usize; 3],
) {
    for row in 0..rows {
        for col in 0..cols {
            let expected = (0..inner)
                .map(|k| lhs[[row, k]] * rhs[[k, col]])
                .sum::<f64>();
            assert_eq!(output[[row, col]], expected, "GEMM at [{row}, {col}]");
        }
    }
}

fn measure([rows, inner, cols]: [usize; 3]) -> Result<(), TensorError> {
    // Dyadic inputs include differences below FP32 precision.
    let lhs = Tensor::from_vec(
        [rows, inner],
        (0..rows * inner)
            .map(|i| (i % 17) as f64 - 8.0 + f64::from((i % 8) as u32) / (1u64 << 30) as f64)
            .collect(),
    )?;
    let rhs = Tensor::from_vec(
        [inner, cols],
        (0..inner * cols)
            .map(|i| ((i % 19) as f64 - 9.0) / 8.0)
            .collect(),
    )?;
    let mut output = Tensor::from_vec([rows, cols], vec![f64::NAN; rows * cols])?;
    gemm(
        lhs.as_slice(),
        rhs.as_slice(),
        output.as_mut_slice(),
        inner,
        cols,
    );
    check(&lhs, &rhs, &output, [rows, inner, cols]);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        gemm(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(output.as_mut_slice()),
            inner,
            cols,
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    check(&lhs, &rhs, &output, [rows, inner, cols]);
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=gemm dtype=f64 m={rows} k={inner} n={cols} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    measure([40, 48, 64])?;
    measure([16, 63, 17])
}
