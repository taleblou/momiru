// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Measures the library's Light canonical FP32 min/max primitives over
//! preallocated host tensors. No native plan dispatch occurs in this loop.
//! The optional linked-release kernel correctness probe is untimed.

use momiru::tensor_light::execution::cpu::{
    builtin_cpu_release_kernel_selfcheck, builtin_cpu_release_profile_matches,
};
use momiru::tensor_light::{
    canonical_maximum, canonical_minimum, Tensor, TensorError, CANONICAL_QUIET_NAN32,
};
use std::{hint::black_box, time::Instant};

fn apply(lhs: &[f32], rhs: &[f32], output: &mut [f32], op: &impl Fn(f32, f32) -> f32) {
    for ((&x, &y), actual) in lhs.iter().zip(rhs).zip(output) {
        *actual = op(x, y);
    }
}

fn verify(lhs: &[f32], rhs: &[f32], output: &[f32], maximum: bool) {
    for (i, ((&x, &y), &actual)) in lhs.iter().zip(rhs).zip(output).enumerate() {
        let expected = if x.is_nan() || y.is_nan() {
            CANONICAL_QUIET_NAN32
        } else if x.total_cmp(&y).is_lt() == maximum {
            y
        } else {
            x
        };
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "canonical comparison at {i}"
        );
    }
}

fn measure(
    lhs: &Tensor<2>,
    rhs: &Tensor<2>,
    name: &str,
    maximum: bool,
    op: impl Fn(f32, f32) -> f32,
) -> Result<(), TensorError> {
    let mut output = Tensor::zeros(*lhs.dims())?;
    apply(lhs.as_slice(), rhs.as_slice(), output.as_mut_slice(), &op);
    verify(lhs.as_slice(), rhs.as_slice(), output.as_slice(), maximum);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        apply(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(output.as_mut_slice()),
            &op,
        );
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(lhs.as_slice(), rhs.as_slice(), output.as_slice(), maximum);
    samples.sort_unstable();
    println!(
        "implementation=light_canonical_primitive operation={name} dtype=f32 elements={} iterations={} median_ns={}",
        lhs.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    if builtin_cpu_release_profile_matches() {
        assert!(
            builtin_cpu_release_kernel_selfcheck(),
            "linked CPU kernel self-check failed"
        );
        println!("check=linked_release_kernel_selfcheck status=PASS");
    } else {
        println!(
            "check=linked_release_kernel_selfcheck status=NOT_RUN reason=release_profile_gate"
        );
    }
    let shape = [128, 257];
    let len = shape[0] * shape[1];
    let mut a: Vec<_> = (0..len).map(|i| (i % 59) as f32 - 29.0).collect();
    let mut b: Vec<_> = (0..len).map(|i| (i % 127) as f32 - 63.0).collect();
    a[..6].copy_from_slice(&[
        -0.0,
        0.0,
        f32::from_bits(0x7fc0_00ab),
        1.0,
        f32::INFINITY,
        f32::from_bits(1),
    ]);
    b[..6].copy_from_slice(&[
        0.0,
        -0.0,
        3.0,
        f32::from_bits(0xffc0_0001),
        f32::NEG_INFINITY,
        0.0,
    ]);
    let lhs = Tensor::<2>::try_from_vec(shape, a)?;
    let rhs = Tensor::<2>::try_from_vec(shape, b)?;
    measure(&lhs, &rhs, "minimum", false, canonical_minimum)?;
    measure(&lhs, &rhs, "maximum", true, canonical_maximum)
}
