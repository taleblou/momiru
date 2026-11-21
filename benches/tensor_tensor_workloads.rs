// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Composes General runtime-rank permutation, reversal, cropping and
//! broadcasting over U64 data. A separate write workload checks an exclusive
//! crop of the same signed-stride mapping.

use momiru::tensor::{Tensor, TensorError};
use std::{hint::black_box, time::Instant};

const BATCH: usize = 2;
const ROWS: usize = 19;
const COLS: usize = 39;
const OUT_ROWS: usize = 16;
const OUT_COLS: usize = 33;
const REPLICAS: usize = 3;
const SOURCE: [usize; 3] = [BATCH, ROWS, COLS];
const OUTPUT: [usize; 4] = [REPLICAS, OUT_COLS, OUT_ROWS, BATCH];
const CROP_START: usize = 2;
const ROW_START: usize = 1;

fn value(flat: usize) -> u64 {
    (1 << 60) + flat as u64 * 7
}

fn gather(input: &Tensor<u64>, output: &mut [u64]) -> Result<(), TensorError> {
    let view = input
        .view()
        .permute([2, 1, 0])?
        .flip(0)?
        .narrow(0, CROP_START, OUT_COLS)?
        .narrow(1, ROW_START, OUT_ROWS)?
        .broadcast_to(OUTPUT)?;
    let mut cells = output.iter_mut();
    for replica in 0..REPLICAS {
        for col in 0..OUT_COLS {
            for row in 0..OUT_ROWS {
                for batch in 0..BATCH {
                    *cells
                        .next()
                        .expect("preallocated output matches view shape") =
                        *view.get([replica, col, row, batch])?;
                }
            }
        }
    }
    Ok(())
}

fn write(output: &mut Tensor<u64>) -> Result<(), TensorError> {
    let mut crop = output
        .view_mut()
        .permute([2, 1, 0])?
        .flip(0)?
        .narrow(0, CROP_START, OUT_COLS)?
        .narrow(1, ROW_START, OUT_ROWS)?;
    for col in 0..OUT_COLS {
        for row in 0..OUT_ROWS {
            for batch in 0..BATCH {
                *crop.get_mut([col, row, batch])? += 1;
            }
        }
    }
    Ok(())
}

fn verify_gather(output: &[u64]) {
    for (i, &actual) in output.iter().enumerate() {
        let batch = i % BATCH;
        let row = i / BATCH % OUT_ROWS + ROW_START;
        let col = COLS - 1 - CROP_START - i / (BATCH * OUT_ROWS) % OUT_COLS;
        assert_eq!(actual, value((batch * ROWS + row) * COLS + col));
    }
}

fn verify_write(output: &[u64]) {
    let first_column = COLS - CROP_START - OUT_COLS;
    let column_end = COLS - CROP_START;
    for (i, &actual) in output.iter().enumerate() {
        let row = i / COLS % ROWS;
        let col = i % COLS;
        let written = (ROW_START..ROW_START + OUT_ROWS).contains(&row)
            && (first_column..column_end).contains(&col);
        assert_eq!(actual, value(i) + u64::from(written));
    }
}

fn report(operation: &str, elements: usize, mut samples: [u128; 9]) {
    samples.sort_unstable();
    println!(
        "implementation=general_host operation={operation} dtype=u64 scope=layout_and_access elements={elements} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
}

fn main() -> Result<(), TensorError> {
    let input = Tensor::from_vec(SOURCE, (0..BATCH * ROWS * COLS).map(value).collect())?;
    let mut gathered = Tensor::from_vec(OUTPUT, vec![u64::MAX; OUTPUT.iter().product()])?;
    let mut modified = input.clone();
    gather(&input, gathered.as_mut_slice())?;
    verify_gather(gathered.as_slice());
    write(&mut modified)?;
    verify_write(modified.as_slice());

    let mut read_times = [0; 9];
    let mut write_times = [0; 9];
    for sample in &mut read_times {
        let start = Instant::now();
        gather(black_box(&input), black_box(gathered.as_mut_slice()))?;
        black_box(gathered.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify_gather(gathered.as_slice());
    for sample in &mut write_times {
        modified.as_mut_slice().copy_from_slice(input.as_slice());
        let start = Instant::now();
        write(black_box(&mut modified))?;
        black_box(modified.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify_write(modified.as_slice());
    report(
        "permute_flip_crop_broadcast_gather",
        gathered.len(),
        read_times,
    );
    report(
        "permute_flip_crop_mutate",
        BATCH * OUT_ROWS * OUT_COLS,
        write_times,
    );
    Ok(())
}
