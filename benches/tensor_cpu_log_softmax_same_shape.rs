// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Stable last-axis log-softmax over General FP32 host tensors. The local
//! reference accumulates in FP64 and writes into a preallocated FP32 output.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 128;
const COLS: usize = 257;
const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);

fn log_softmax(input: &[f32], output: &mut [f32]) {
    for (row, out) in input
        .as_chunks::<COLS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<COLS>().0)
    {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if row.iter().any(|x| x.is_nan()) || maximum == f32::NEG_INFINITY {
            out.fill(CANONICAL_NAN);
        } else if maximum == f32::INFINITY {
            let positives = row.iter().filter(|&&x| x == f32::INFINITY).count();
            let shared_log_mass = -(positives as f64).ln() as f32;
            for (&x, result) in row.iter().zip(out) {
                *result = if x == f32::INFINITY {
                    shared_log_mass
                } else {
                    f32::NEG_INFINITY
                };
            }
        } else {
            let maximum = f64::from(maximum);
            let log_total = row
                .iter()
                .map(|&x| (f64::from(x) - maximum).exp())
                .sum::<f64>()
                .ln();
            for (&x, result) in row.iter().zip(out) {
                *result = (f64::from(x) - maximum - log_total) as f32;
            }
        }
    }
}

fn verify(input: &[f32], output: &[f32]) {
    let inputs = input.as_chunks::<COLS>().0;
    let results = output.as_chunks::<COLS>().0;
    for row in &results[..2] {
        assert!(row.iter().all(|x| x.to_bits() == CANONICAL_NAN.to_bits()));
    }
    for (column, &value) in results[2].iter().enumerate() {
        let expected = if column == 0 || column == COLS - 1 {
            -std::f32::consts::LN_2
        } else {
            f32::NEG_INFINITY
        };
        assert_eq!(value, expected);
    }
    for (row, out) in inputs[3..].iter().zip(&results[3..]) {
        // Independent sequential log-add-exp reference. This does not use
        // the candidate's row maximum and exp-sum reduction.
        let normalizer = row
            .iter()
            .copied()
            .filter(|&x| x != f32::NEG_INFINITY)
            .map(f64::from)
            .reduce(|a, b| a.max(b) + (-(a - b).abs()).exp().ln_1p())
            .unwrap();
        for (&x, &y) in row.iter().zip(out) {
            if x == f32::NEG_INFINITY {
                assert_eq!(y, f32::NEG_INFINITY);
            } else {
                let expected = f64::from(x) - normalizer;
                let error = (f64::from(y) - expected).abs();
                assert!(
                    error <= 1.0e-6 + 1.0e-7 * expected.abs(),
                    "log-softmax of {x}: {y} vs {expected}"
                );
            }
        }
    }
}

fn main() -> Result<(), TensorError> {
    let mut logits: Vec<_> = (0..ROWS * COLS)
        .map(|i| ((i % 641) as f32 - 320.0) * 3.0)
        .collect();
    logits[0] = f32::from_bits(0x7fc0_0189);
    logits[COLS..2 * COLS].fill(f32::NEG_INFINITY);
    logits[2 * COLS] = f32::INFINITY;
    logits[3 * COLS - 1] = f32::INFINITY;
    logits[3 * COLS] = f32::NEG_INFINITY;
    let input = Tensor::from_vec([ROWS, COLS], logits)?;
    let mut output = Tensor::from_vec([ROWS, COLS], vec![f32::NAN; ROWS * COLS])?;
    log_softmax(input.as_slice(), output.as_mut_slice());
    verify(input.as_slice(), output.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        log_softmax(
            black_box(input.as_slice()),
            black_box(output.as_mut_slice()),
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(input.as_slice(), output.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=last_axis_log_softmax dtype=f32 shape={ROWS}x{COLS} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}
