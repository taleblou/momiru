// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Checked dense and signed-view reads through Light's public APIs. Coordinate
//! generation, view construction, and validation are outside the timed loop.

use momiru::tensor_light::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 128;
const COLS: usize = 257;
const QUERIES: usize = 8_193;

fn value(flat: usize) -> i32 {
    (flat % 1009) as i32 - 504
}

fn validate(
    queries: &[[usize; 2]],
    read: &impl Fn([usize; 2]) -> Result<f32, TensorError>,
    reversed: bool,
) -> Result<f64, TensorError> {
    let mut sum = 0_i64;
    for &[row, col] in queries {
        let physical_col = if reversed { COLS - 1 - col } else { col };
        let expected = value(row * COLS + physical_col);
        assert_eq!(read([row, col])?, expected as f32);
        sum += i64::from(expected);
    }
    Ok(sum as f64)
}

fn read_sum(
    queries: &[[usize; 2]],
    read: &impl Fn([usize; 2]) -> Result<f32, TensorError>,
) -> Result<f64, TensorError> {
    let mut sum = 0.0;
    for &index in queries {
        sum += f64::from(read(index)?);
    }
    Ok(sum)
}

fn measure(
    operation: &str,
    queries: &[[usize; 2]],
    read: impl Fn([usize; 2]) -> Result<f32, TensorError>,
    reversed: bool,
) -> Result<(), TensorError> {
    let expected = validate(queries, &read, reversed)?;
    assert_eq!(read_sum(queries, &read)?, expected);
    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        let checksum = black_box(read_sum(black_box(queries), &read)?);
        *sample = start.elapsed().as_nanos();
        assert_eq!(checksum, expected);
    }
    validate(queries, &read, reversed)?;
    samples.sort_unstable();
    println!(
        "implementation=light_host operation={operation} dtype=f32 queries={} iterations={} median_ns={}",
        queries.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::<2>::try_from_vec(
        [ROWS, COLS],
        (0..ROWS * COLS).map(|i| value(i) as f32).collect(),
    )?;
    let queries: Vec<_> = (0..QUERIES)
        .map(|i| [(i * 17 + 3) % ROWS, (i * 43 + 11) % COLS])
        .collect();
    let flipped = tensor.signed_view()?.flip(1)?;
    measure(
        "dense_checked_read",
        &queries,
        |index| black_box(&tensor).get(index),
        false,
    )?;
    measure(
        "negative_stride_checked_read",
        &queries,
        |index| black_box(&flipped).get(index),
        true,
    )
}
