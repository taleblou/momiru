// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Benchmark-local portable and explicit AVX FP32 addition over General
//! host tensors. The portable source may be auto-vectorized by the compiler.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

fn portable_add(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    for ((&x, &y), out) in lhs.iter().zip(rhs).zip(output) {
        *out = x + y;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_avx(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let (left, left_tail) = lhs.as_chunks::<8>();
    let (right, right_tail) = rhs.as_chunks::<8>();
    let (chunks, tail) = output.as_chunks_mut::<8>();
    for ((x, y), out) in left.iter().zip(right).zip(chunks) {
        // SAFETY: all three arrays hold eight live elements. The unaligned
        // forms accept normal f32 alignment, and &mut excludes input aliasing.
        unsafe {
            let sum = _mm256_add_ps(_mm256_loadu_ps(x.as_ptr()), _mm256_loadu_ps(y.as_ptr()));
            _mm256_storeu_ps(out.as_mut_ptr(), sum);
        }
    }
    portable_add(left_tail, right_tail, tail);
}

fn left_value(index: usize) -> i32 {
    index as i32 % 101 - 50
}

fn right_value(index: usize) -> i32 {
    index as i32 % 67 - 33
}

fn verify(output: &[f32]) {
    for (index, &actual) in output.iter().enumerate() {
        assert_eq!(
            actual,
            (left_value(index) + right_value(index)) as f32,
            "element {index}"
        );
    }
}

fn measure(
    name: &str,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    kernel: impl Fn(&[f32], &[f32], &mut [f32]),
) {
    output.fill(f32::NAN);
    kernel(lhs, rhs, output);
    verify(output);
    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        kernel(black_box(lhs), black_box(rhs), black_box(output));
        black_box(&*output);
        *sample = start.elapsed().as_nanos();
    }
    verify(output);
    samples.sort_unstable();
    println!(
        "implementation={name} operation=add dtype=f32 elements={} iterations={} median_ns={}",
        output.len(),
        samples.len(),
        samples[samples.len() / 2]
    );
}

fn main() -> Result<(), TensorError> {
    // A nonmultiple of eight checks the vector path's scalar tail.
    const ELEMENTS: usize = 32_771;
    let lhs = Tensor::from_vec(
        [ELEMENTS],
        (0..ELEMENTS).map(|i| left_value(i) as f32).collect(),
    )?;
    let rhs = Tensor::from_vec(
        [ELEMENTS],
        (0..ELEMENTS).map(|i| right_value(i) as f32).collect(),
    )?;
    let mut output = Tensor::from_vec([ELEMENTS], vec![f32::NAN; ELEMENTS])?;
    measure(
        "portable_reference",
        lhs.as_slice(),
        rhs.as_slice(),
        output.as_mut_slice(),
        portable_add,
    );

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        measure(
            "explicit_avx_reference",
            lhs.as_slice(),
            rhs.as_slice(),
            output.as_mut_slice(),
            |a, b, out| {
                // SAFETY: runtime detection confirms AVX. All arrays were
                // allocated with the same element count, and output is exclusive.
                unsafe { add_avx(a, b, out) };
            },
        );
        return Ok(());
    }
    println!("implementation=explicit_avx_reference operation=add status=UNSUPPORTED reason=avx_unavailable");
    Ok(())
}
