// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Scalar FP32 addition over General host tensors. The timed region uses
//! preallocated slices and excludes input generation and output validation.

use momiru::tensor::Tensor;
use std::hint::black_box;
use std::time::Instant;

fn scalar_add(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    for index in 0..out.len() {
        out[index] = lhs[index] + rhs[index];
    }
}

fn validate(output: &[f32]) {
    for (index, &value) in output.iter().enumerate() {
        let expected = (index % 257 + index % 127) as i32 - 191;
        assert_eq!(value, expected as f32, "addition at index {index}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cargo may append --bench when launching a harness-free target.
    let mut args = std::env::args().skip(1).filter(|arg| arg != "--bench");
    let elements: usize = args.next().as_deref().unwrap_or("8192").parse()?;
    if args.next().is_some() || !(1..=16_777_216).contains(&elements) {
        return Err("usage: tensor_cpu_baseline [elements in 1..=16777216]".into());
    }

    let lhs = Tensor::from_vec(
        [elements],
        (0..elements)
            .map(|i| ((i % 257) as i32 - 128) as f32)
            .collect(),
    )?;
    let rhs = Tensor::from_vec(
        [elements],
        (0..elements)
            .map(|i| ((i % 127) as i32 - 63) as f32)
            .collect(),
    )?;
    let mut out = Tensor::from_vec([elements], vec![f32::NAN; elements])?;

    scalar_add(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice());
    validate(out.as_slice());

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        scalar_add(
            black_box(lhs.as_slice()),
            black_box(rhs.as_slice()),
            black_box(out.as_mut_slice()),
        );
        black_box(out.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    validate(out.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=scalar_reference operation=add dtype=f32 elements={elements} iterations={} median_ns={}",
        samples.len(),
        samples[samples.len() / 2]
    );
    Ok(())
}
