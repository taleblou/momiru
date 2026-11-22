// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Scalar row/column sums followed by broadcasting into an existing output
//! with the same shape. Timing includes both reduction and output writes.

use momiru::tensor::{Tensor, TensorError};
use std::hint::black_box;
use std::time::Instant;

const ROWS: usize = 96;
const COLS: usize = 160;

fn row_sum_broadcast(input: &[f32], output: &mut [f32]) {
    for (row, out) in input
        .as_chunks::<COLS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<COLS>().0)
    {
        let sum = row.iter().copied().sum();
        out.fill(sum);
    }
}

fn column_sum_broadcast(input: &[f32], output: &mut [f32]) {
    let (sums, remainder) = output.split_at_mut(COLS);
    sums.fill(0.0);
    for row in input.as_chunks::<COLS>().0 {
        for (sum, value) in sums.iter_mut().zip(row) {
            *sum += value;
        }
    }
    for row in remainder.as_chunks_mut::<COLS>().0 {
        row.copy_from_slice(sums);
    }
}

fn value(row: usize, col: usize) -> i32 {
    ((row * 3 + col * 5) % 37) as i32 - 18
}

fn measure(
    input: &Tensor<f32>,
    name: &str,
    kernel: impl Fn(&[f32], &mut [f32]),
    oracle: impl Fn(usize, usize) -> i32,
) -> Result<(), TensorError> {
    let mut output = Tensor::from_vec(input.shape(), vec![f32::NAN; input.len()])?;
    let check = |out: &[f32]| {
        for (row, values) in out.as_chunks::<COLS>().0.iter().enumerate() {
            for (col, &result) in values.iter().enumerate() {
                assert_eq!(result, oracle(row, col) as f32, "{name} at [{row}, {col}]");
            }
        }
    };
    kernel(input.as_slice(), output.as_mut_slice());
    check(output.as_slice());
    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        kernel(
            black_box(input.as_slice()),
            black_box(output.as_mut_slice()),
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    check(output.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation={name} dtype=f32 shape={ROWS}x{COLS} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let input = Tensor::from_vec(
        [ROWS, COLS],
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| value(row, col) as f32))
            .collect(),
    )?;
    measure(&input, "row_sum_broadcast", row_sum_broadcast, |row, _| {
        (0..COLS).map(|col| value(row, col)).sum()
    })?;
    measure(
        &input,
        "column_sum_broadcast",
        column_sum_broadcast,
        |_, col| (0..ROWS).map(|row| value(row, col)).sum(),
    )
}
