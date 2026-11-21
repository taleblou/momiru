// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Host allocation and initialization for the two independent tensor APIs.
//! Includes the constructor and any source Vec creation. Validation, freeing
//! storage and any page touches caused by validation are outside timing.

use momiru::{tensor, tensor_light};
use std::{hint::black_box, time::Instant};

const SHAPE: [usize; 2] = [128, 257];
const ELEMENTS: usize = SHAPE[0] * SHAPE[1];
const ALIGNMENT: usize = 256;
const FILL: f32 = -7.25;

fn measure<T, E>(
    name: &str,
    operation: &str,
    create: impl Fn() -> Result<T, E>,
    verify: impl Fn(&T),
) -> Result<(), E> {
    verify(&create()?);
    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        let result = create()?;
        black_box(&result);
        *sample = start.elapsed().as_nanos();
        verify(&result);
        drop(result);
    }
    samples.sort_unstable();
    println!(
        "implementation={name} operation={operation} dtype=f32 elements={ELEMENTS} scope=allocation_and_initialization iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn check_light(output: &tensor_light::Tensor<2>, expected: f32) {
    assert_eq!(output.dims(), &SHAPE);
    assert_eq!(output.alignment(), ALIGNMENT);
    assert_eq!(output.as_ptr().addr() % ALIGNMENT, 0);
    assert!(output
        .as_slice()
        .iter()
        .all(|value| value.to_bits() == expected.to_bits()));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    measure(
        "general_host",
        "from_filled_vec",
        || {
            let data = vec![black_box(FILL); ELEMENTS];
            tensor::Tensor::from_vec(black_box(SHAPE), data)
        },
        |output| {
            assert_eq!(output.shape(), &SHAPE);
            assert!(output
                .as_slice()
                .iter()
                .all(|value| value.to_bits() == FILL.to_bits()));
        },
    )?;
    measure(
        "light_host",
        "full_aligned",
        || tensor_light::Tensor::full_aligned(black_box(SHAPE), black_box(FILL), ALIGNMENT),
        |output| check_light(output, FILL),
    )?;
    measure(
        "light_host",
        "zeros_aligned",
        || tensor_light::Tensor::zeros_aligned(black_box(SHAPE), ALIGNMENT),
        |output| check_light(output, 0.0),
    )?;
    Ok(())
}
