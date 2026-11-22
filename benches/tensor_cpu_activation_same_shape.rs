// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Allocation-free scalar ReLU and sigmoid references over General host tensors.

use momiru::tensor::{Tensor, TensorError};
use std::hint::black_box;
use std::time::Instant;

fn relu(value: f32) -> f32 {
    if value.is_nan() || value > 0.0 {
        value
    } else {
        0.0
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn activate(input: &[f32], output: &mut [f32], operation: impl Fn(f32) -> f32) {
    for (value, result) in input.iter().zip(output) {
        *result = operation(*value);
    }
}

fn measure(
    name: &str,
    input: &Tensor<f32>,
    operation: impl Fn(f32) -> f32,
    validate: impl Fn(f32, f32),
) -> Result<(), TensorError> {
    let mut output = Tensor::from_vec(input.shape(), vec![f32::NAN; input.len()])?;
    let verify = |result: &[f32]| {
        for (&x, &y) in input.as_slice().iter().zip(result) {
            validate(x, y);
        }
    };
    activate(input.as_slice(), output.as_mut_slice(), &operation);
    verify(output.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        activate(
            black_box(input.as_slice()),
            black_box(output.as_mut_slice()),
            &operation,
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(output.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation={name} dtype=f32 elements={} iterations={} median_ns={}",
        input.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let mut data: Vec<_> = (0..128 * 256)
        .map(|index| ((index % 1025) as f32 - 512.0) * 0.125)
        .collect();
    data[..6].copy_from_slice(&[
        -0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        -100.0,
    ]);
    let input = Tensor::from_vec([128, 256], data)?;

    measure("relu", &input, relu, |x, y| {
        if x.is_nan() {
            assert!(y.is_nan());
        } else {
            let expected = if x <= 0.0 { 0.0 } else { x };
            assert_eq!(y.to_bits(), expected.to_bits());
        }
    })?;
    measure("sigmoid", &input, sigmoid, |x, y| {
        if x.is_nan() {
            assert!(y.is_nan());
        } else {
            let expected = 1.0 / (1.0 + (-f64::from(x)).exp());
            assert!(
                (f64::from(y) - expected).abs() <= 2.0e-7,
                "sigmoid({x}) = {y}"
            );
        }
    })
}
