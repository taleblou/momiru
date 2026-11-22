// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Benchmark-local stable same-shape softplus and log-add-exp references.
//! Inputs and outputs are General FP32 host storage. Oracles use direct
//! FP64 formulas over a finite input range where the exponentials fit.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

fn softplus(x: f32, _: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn logaddexp(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    let peak = x.max(y);
    if peak.is_infinite() {
        return peak;
    }
    peak + (-(x - y).abs()).exp().ln_1p()
}

fn apply(lhs: &[f32], rhs: &[f32], out: &mut [f32], op: &impl Fn(f32, f32) -> f32) {
    for ((&a, &b), result) in lhs.iter().zip(rhs).zip(out) {
        *result = op(a, b);
    }
}

fn measure(
    name: &str,
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    op: impl Fn(f32, f32) -> f32,
    oracle: impl Fn(f64, f64) -> f64,
) -> Result<(), TensorError> {
    let mut out = Tensor::from_vec(lhs.shape(), vec![f32::NAN; lhs.len()])?;
    let verify = |results: &[f32]| {
        for ((&a, &b), &actual) in lhs.as_slice().iter().zip(rhs.as_slice()).zip(results) {
            let expected = oracle(f64::from(a), f64::from(b)) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else if expected.is_infinite() {
                assert_eq!(actual, expected);
            } else {
                let tolerance = 2.0e-7 * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance, "{name}({a}, {b})");
                if name == "softplus" && a < -20.0 && expected > 0.0 {
                    let tail_tolerance = 3.0e-7 * expected + f32::from_bits(2);
                    assert!(
                        actual > 0.0 && (actual - expected).abs() <= tail_tolerance,
                        "softplus tail at {a}"
                    );
                }
            }
        }
    };
    apply(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice(), &op);
    verify(out.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        apply(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(out.as_mut_slice()),
            &op,
        );
        black_box(out.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(out.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation={name} dtype=f32 elements={} iterations={} median_ns={}",
        lhs.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let shape = [128, 257];
    let len = shape[0] * shape[1];
    let mut a: Vec<_> = (0..len).map(|i| (i % 481) as f32 * 0.5 - 120.0).collect();
    let mut b: Vec<_> = (0..len).map(|i| (i % 337) as f32 * 0.5 - 84.0).collect();
    a[..6].copy_from_slice(&[
        -90.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        100.0,
        -50.0,
    ]);
    b[..6].copy_from_slice(&[-91.0, f32::NEG_INFINITY, f32::INFINITY, 0.0, 99.0, f32::NAN]);
    let lhs = Tensor::from_vec(shape, a)?;
    let rhs = Tensor::from_vec(shape, b)?;
    measure("softplus", &lhs, &rhs, softplus, |x, _| x.exp().ln_1p())?;
    measure("logaddexp", &lhs, &rhs, logaddexp, |x, y| {
        (x.exp() + y.exp()).ln()
    })
}
