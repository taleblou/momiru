// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! FP32 reference matrix multiplication over General host tensors with
//! row-major and physically transposed RHS inputs.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

fn matmul<const RHS_TRANSPOSED: bool>(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    [rows, inner, cols]: [usize; 3],
) {
    for row in 0..rows {
        let result = &mut output[row * cols..(row + 1) * cols];
        result.fill(0.0);
        for k in 0..inner {
            let left = lhs[row * inner + k];
            for (col, value) in result.iter_mut().enumerate() {
                let right = if RHS_TRANSPOSED {
                    rhs[col * inner + k]
                } else {
                    rhs[k * cols + col]
                };
                *value += left * right;
            }
        }
    }
}

fn left_value(row: usize, k: usize) -> i32 {
    ((row * 7 + k * 5) % 31) as i32 - 15
}

fn right_value(k: usize, col: usize) -> i32 {
    ((k * 11 + col * 3) % 29) as i32 - 14
}

fn verify(output: &Tensor<f32>, [rows, inner, cols]: [usize; 3]) {
    assert_eq!(output.shape(), &[rows, cols]);
    for row in 0..rows {
        for col in 0..cols {
            let expected = (0..inner)
                .map(|k| left_value(row, k) * right_value(k, col))
                .sum::<i32>();
            assert_eq!(
                output[[row, col]],
                expected as f32,
                "GEMM at [{row}, {col}]"
            );
        }
    }
}

fn measure<const RHS_TRANSPOSED: bool>(
    shape @ [rows, inner, cols]: [usize; 3],
) -> Result<(), TensorError> {
    let lhs = Tensor::from_vec(
        [rows, inner],
        (0..rows)
            .flat_map(|row| (0..inner).map(move |k| left_value(row, k) as f32))
            .collect(),
    )?;
    let rhs = if RHS_TRANSPOSED {
        Tensor::from_vec(
            [cols, inner],
            (0..cols)
                .flat_map(|col| (0..inner).map(move |k| right_value(k, col) as f32))
                .collect(),
        )?
    } else {
        Tensor::from_vec(
            [inner, cols],
            (0..inner)
                .flat_map(|k| (0..cols).map(move |col| right_value(k, col) as f32))
                .collect(),
        )?
    };
    let mut output = Tensor::from_vec([rows, cols], vec![f32::NAN; rows * cols])?;
    matmul::<RHS_TRANSPOSED>(lhs.as_slice(), rhs.as_slice(), output.as_mut_slice(), shape);
    verify(&output, shape);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        matmul::<RHS_TRANSPOSED>(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(output.as_mut_slice()),
            shape,
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(&output, shape);
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=matmul dtype=f32 m={rows} k={inner} n={cols} rhs_transposed={RHS_TRANSPOSED} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let mut empty_contraction = [f32::NAN; 6];
    matmul::<false>(&[], &[], &mut empty_contraction, [2, 0, 3]);
    assert_eq!(empty_contraction.map(f32::to_bits), [0; 6]);
    measure::<false>([47, 91, 65])?;
    measure::<true>([47, 91, 65])
}
