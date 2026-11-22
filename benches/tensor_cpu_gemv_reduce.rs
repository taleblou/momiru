// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Scalar FP32 matrix-vector multiplication followed by a sum over the result.
//! The timed path materializes the vector into preallocated General host storage.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 192;
const COLS: usize = 257;

fn matrix_value(row: usize, col: usize) -> i32 {
    ((row * 7 + col * 3) % 31) as i32 - 15
}

fn vector_value(col: usize) -> i32 {
    (col % 11) as i32 - 5
}

fn gemv_reduce(matrix: &[f32], vector: &[f32], output: &mut [f32]) -> f32 {
    for (row, result) in matrix.as_chunks::<COLS>().0.iter().zip(output.iter_mut()) {
        *result = row.iter().zip(vector).map(|(&a, &x)| a * x).sum();
    }
    output.iter().sum()
}

fn verify(output: &[f32], total: f32) {
    let mut expected_total = 0;
    for (row, &actual) in output.iter().enumerate() {
        let expected = (0..COLS)
            .map(|col| matrix_value(row, col) * vector_value(col))
            .sum::<i32>();
        assert_eq!(actual, expected as f32, "GEMV row {row}");
        expected_total += expected;
    }
    assert_eq!(total, expected_total as f32, "sum over GEMV output");
}

fn main() -> Result<(), TensorError> {
    let matrix = Tensor::from_vec(
        [ROWS, COLS],
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| matrix_value(row, col) as f32))
            .collect(),
    )?;
    let vector = Tensor::from_vec(
        [COLS],
        (0..COLS).map(|col| vector_value(col) as f32).collect(),
    )?;
    let mut output = Tensor::from_vec([ROWS], vec![f32::NAN; ROWS])?;
    let mut total = gemv_reduce(matrix.as_slice(), vector.as_slice(), output.as_mut_slice());
    verify(output.as_slice(), total);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        total = gemv_reduce(
            black_box(matrix.as_slice()),
            black_box(vector.as_slice()),
            black_box(output.as_mut_slice()),
        );
        black_box((total, output.as_slice()));
        *sample = start.elapsed().as_nanos();
    }
    verify(output.as_slice(), total);
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=gemv_then_sum dtype=f32 shape={ROWS}x{COLS} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}
