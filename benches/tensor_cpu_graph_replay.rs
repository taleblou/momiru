// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Benchmark-local replay of a fixed add, ReLU, and scaled residual graph.
//! The timed region dispatches three preselected stages over two reusable
//! General host buffers. It uses no frozen plan, native artifact, or AOT evidence.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

type Stage = fn(&[f32], &[f32], &mut [f32]);

fn add(input: &[f32], residual: &[f32], output: &mut [f32]) {
    for ((&x, &r), out) in input.iter().zip(residual).zip(output) {
        *out = x + r;
    }
}

fn relu(input: &[f32], _: &[f32], output: &mut [f32]) {
    for (&x, out) in input.iter().zip(output) {
        *out = x.max(0.0);
    }
}

fn scaled_residual(input: &[f32], residual: &[f32], output: &mut [f32]) {
    for ((&x, &r), out) in input.iter().zip(residual).zip(output) {
        *out = 2.0 * x + r;
    }
}

struct Replay {
    stages: [Stage; 3],
    first: Tensor<f32>,
    second: Tensor<f32>,
}

impl Replay {
    fn run(&mut self, input: &[f32], residual: &[f32]) -> &[f32] {
        let [first, second, third] = self.stages;
        first(input, residual, self.first.as_mut_slice());
        second(self.first.as_slice(), residual, self.second.as_mut_slice());
        third(self.second.as_slice(), residual, self.first.as_mut_slice());
        self.first.as_slice()
    }
}

fn input_value(index: usize) -> i32 {
    (index % 113) as i32 - 56
}

fn residual_value(index: usize) -> i32 {
    (index % 17) as i32 - 8
}

fn verify(input: &[f32], residual: &[f32], output: &[f32]) {
    for ((&x, &r), &actual) in input.iter().zip(residual).zip(output) {
        let expected = 2 * (x as i32 + r as i32).max(0) + r as i32;
        assert_eq!(actual, expected as f32);
    }
}

fn main() -> Result<(), TensorError> {
    const ELEMENTS: usize = 32_771;
    let first_input = Tensor::from_vec(
        [ELEMENTS],
        (0..ELEMENTS).map(|i| input_value(i) as f32).collect(),
    )?;
    let second_input = Tensor::from_vec(
        [ELEMENTS],
        (0..ELEMENTS).map(|i| -input_value(i) as f32).collect(),
    )?;
    let residual = Tensor::from_vec(
        [ELEMENTS],
        (0..ELEMENTS).map(|i| residual_value(i) as f32).collect(),
    )?;
    let mut replay = Replay {
        stages: [add, relu, scaled_residual],
        first: Tensor::from_vec([ELEMENTS], vec![f32::NAN; ELEMENTS])?,
        second: Tensor::from_vec([ELEMENTS], vec![f32::NAN; ELEMENTS])?,
    };
    let r = residual.as_slice();
    // Exercise consecutive runs with different data before measuring replay.
    for input in [first_input.as_slice(), second_input.as_slice()] {
        verify(input, r, replay.run(input, r));
    }

    let input = first_input.as_slice();
    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        black_box(replay.run(black_box(input), black_box(r)));
        *sample = start.elapsed().as_nanos();
    }
    verify(input, r, replay.first.as_slice());
    samples.sort_unstable();
    println!(
        "implementation=local_reference_replay operation=add_relu_scaled_residual dtype=f32 elements={ELEMENTS} stages=3 iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}
