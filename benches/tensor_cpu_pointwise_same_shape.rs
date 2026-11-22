// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Scalar FP32 multiply and maximum over equal-shape General host tensors.
//! Both operations reuse output storage and validate IEEE edge cases.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);

fn maximum(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() || rhs.is_nan() {
        CANONICAL_NAN
    } else if lhs == 0.0 && rhs == 0.0 {
        f32::from_bits(lhs.to_bits() & rhs.to_bits())
    } else {
        lhs.max(rhs)
    }
}

fn pointwise(lhs: &[f32], rhs: &[f32], output: &mut [f32], kernel: &impl Fn(f32, f32) -> f32) {
    for ((&a, &b), out) in lhs.iter().zip(rhs).zip(output) {
        *out = kernel(a, b);
    }
}

fn measure(
    name: &str,
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    kernel: impl Fn(f32, f32) -> f32,
    verify: impl Fn(f32, f32, f32),
) -> Result<(), TensorError> {
    assert_eq!(lhs.shape(), rhs.shape());
    let mut output = Tensor::from_vec(lhs.shape(), vec![f32::NAN; lhs.len()])?;
    let check = |out: &[f32]| {
        for ((&a, &b), &actual) in lhs.as_slice().iter().zip(rhs.as_slice()).zip(out) {
            verify(a, b, actual);
        }
    };
    pointwise(
        lhs.as_slice(),
        rhs.as_slice(),
        output.as_mut_slice(),
        &kernel,
    );
    check(output.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        pointwise(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(output.as_mut_slice()),
            &kernel,
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    check(output.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation={name} dtype=f32 elements={} iterations={} median_ns={}",
        lhs.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let shape = [127, 259];
    let len = shape[0] * shape[1];
    let mut a: Vec<_> = (0..len).map(|i| ((i % 113) as f32 - 56.0) / 8.0).collect();
    let mut b: Vec<_> = (0..len).map(|i| ((i % 67) as f32 - 33.0) / 16.0).collect();
    a[..6].copy_from_slice(&[
        -0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
    ]);
    b[..6].copy_from_slice(&[0.0, -0.0, 1.0, 0.0, 1.0, 0.125]);
    let lhs = Tensor::from_vec(shape, a)?;
    let rhs = Tensor::from_vec(shape, b)?;
    measure(
        "multiply",
        &lhs,
        &rhs,
        |x, y| x * y,
        |x, y, actual| {
            let expected = (f64::from(x) * f64::from(y)) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        },
    )?;
    measure("maximum", &lhs, &rhs, maximum, |x, y, actual| {
        let expected = if x.is_nan() || y.is_nan() {
            CANONICAL_NAN
        } else {
            // total_cmp orders negative zero below positive zero.
            match x.total_cmp(&y) {
                std::cmp::Ordering::Less => y,
                _ => x,
            }
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    })
}
