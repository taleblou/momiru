// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Last-axis log-sum-exp followed by same-shape broadcast over General FP32
//! storage. Timing includes reduction and writes to preallocated output.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 96;
const COLS: usize = 257;

fn logsumexp_broadcast(input: &[f32], output: &mut [f32]) {
    for (row, out) in input
        .as_chunks::<COLS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<COLS>().0)
    {
        let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let result = if row.iter().any(|value| value.is_nan()) {
            f32::NAN
        } else if peak.is_infinite() {
            peak
        } else {
            let peak = f64::from(peak);
            let weights = row
                .iter()
                .map(|&value| (f64::from(value) - peak).exp())
                .sum::<f64>();
            (peak + weights.ln()) as f32
        };
        out.fill(result);
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}

fn verify(input: &[f32], output: &[f32]) {
    for (index, (row, result)) in input
        .as_chunks::<COLS>()
        .0
        .iter()
        .zip(output.as_chunks::<COLS>().0)
        .enumerate()
    {
        let expected = match index {
            0 => f32::NAN,
            1 => f32::NEG_INFINITY,
            2 => f32::INFINITY,
            // A pairwise log-add reference uses neither the maximum over the
            // whole row nor the candidate's reduction over exponentials.
            _ => row
                .iter()
                .copied()
                .map(f64::from)
                .fold(f64::NEG_INFINITY, log_add) as f32,
        };
        for &actual in result {
            if expected.is_nan() {
                assert!(actual.is_nan(), "row {index}");
            } else if expected.is_infinite() {
                assert_eq!(actual, expected, "row {index}");
            } else {
                let tolerance = 2.0e-7 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "row {index}: {actual} vs {expected}"
                );
            }
        }
    }
}

fn main() -> Result<(), TensorError> {
    let mut values: Vec<_> = (0..ROWS * COLS)
        .map(|i| 10_000.0 + ((i % 521) as f32 - 260.0) * 1.5)
        .collect();
    values[0] = f32::NAN;
    values[COLS..2 * COLS].fill(f32::NEG_INFINITY);
    values[2 * COLS] = f32::INFINITY;
    values[3 * COLS] = f32::NEG_INFINITY;
    let input = Tensor::from_vec([ROWS, COLS], values)?;
    let mut output = Tensor::from_vec([ROWS, COLS], vec![f32::NAN; ROWS * COLS])?;
    logsumexp_broadcast(input.as_slice(), output.as_mut_slice());
    verify(input.as_slice(), output.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        logsumexp_broadcast(
            black_box(input.as_slice()),
            black_box(output.as_mut_slice()),
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(input.as_slice(), output.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=last_axis_logsumexp_broadcast dtype=f32 shape={ROWS}x{COLS} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}
