// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-License-Identifier: MIT

//! Composed Light views using reversal, stepped slicing, permutation, and
//! broadcasting, plus exclusive writes through a reversed stepped view.
//! Each timed path includes view construction and indexed data access.

use momiru::tensor_light::{SignedSliceSpec, Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 48;
const COLS: usize = 65;
const SELECTED: usize = COLS / 2;
const BATCHES: usize = 3;
const STEP: SignedSliceSpec = SignedSliceSpec::new(Some(1), None, 2);

fn value(index: usize) -> f32 {
    (index % 509) as f32 - 254.0
}

fn read_workload(input: &Tensor<2>, result: &mut [f32]) -> Result<(), TensorError> {
    let transformed = input
        .signed_view()?
        .flip(1)?
        .slice_axis(1, STEP)?
        .permute([1, 0])?
        .unsqueeze::<3>(0)?
        .broadcast_to([BATCHES, SELECTED, ROWS])?;
    for batch in 0..BATCHES {
        for col in 0..SELECTED {
            for row in 0..ROWS {
                result[(batch * SELECTED + col) * ROWS + row] =
                    transformed.get([batch, col, row])?;
            }
        }
    }
    Ok(())
}

fn write_workload(input: &mut Tensor<2>) -> Result<(), TensorError> {
    let mut view = input
        .signed_view_mut()?
        .flip_mut(1)?
        .slice_axis_mut(1, STEP)?;
    for row in 0..ROWS {
        for col in 0..SELECTED {
            *view.try_get_mut([row, col])? += 1.0;
        }
    }
    Ok(())
}

fn verify_read(result: &[f32]) {
    for (i, &actual) in result.iter().enumerate() {
        let row = i % ROWS;
        let logical_col = (i / ROWS) % SELECTED;
        let physical_col = COLS - 2 - 2 * logical_col;
        assert_eq!(actual, value(row * COLS + physical_col));
    }
}

fn verify_write(output: &Tensor<2>) {
    for (i, &actual) in output.as_slice().iter().enumerate() {
        let written = !(i % COLS).is_multiple_of(2);
        assert_eq!(actual, value(i) + if written { 1.0 } else { 0.0 });
    }
}

fn report(operation: &str, elements: usize, mut samples: [u128; 9]) {
    samples.sort_unstable();
    println!(
        "implementation=light_host operation={operation} dtype=f32 elements={elements} scope=layout_and_access iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
}

fn main() -> Result<(), TensorError> {
    let input = Tensor::<2>::try_from_vec([ROWS, COLS], (0..ROWS * COLS).map(value).collect())?;
    let mut read_output = Tensor::<3>::zeros([BATCHES, SELECTED, ROWS])?;
    let mut write_output = input.try_clone()?;
    read_workload(&input, read_output.as_mut_slice())?;
    verify_read(read_output.as_slice());
    write_workload(&mut write_output)?;
    verify_write(&write_output);

    let mut read_times = [0; 9];
    let mut write_times = [0; 9];
    for sample in &mut read_times {
        let start = Instant::now();
        read_workload(black_box(&input), black_box(read_output.as_mut_slice()))?;
        black_box(read_output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify_read(read_output.as_slice());
    for sample in &mut write_times {
        write_output.copy_from_tensor(&input)?;
        let start = Instant::now();
        write_workload(black_box(&mut write_output))?;
        black_box(write_output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify_write(&write_output);
    report(
        "flip_slice_permute_broadcast_read",
        read_output.len(),
        read_times,
    );
    report("flip_slice_mutate", ROWS * SELECTED, write_times);
    Ok(())
}
