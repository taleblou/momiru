// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! General runtime-rank traversal of a prebuilt reversed view. The rank
//! sweep keeps the first and last extents fixed and inserts singleton axes.

use momiru::tensor::{Tensor, TensorError, TensorView};
use std::{hint::black_box, time::Instant};

fn value(flat: usize) -> f64 {
    (flat % 8191) as f64 / 16.0 + (flat % 13) as f64 / (1u64 << 36) as f64
}

fn materialize(
    view: &TensorView<'_, f64>,
    index: &mut [usize],
    output: &mut [f64],
    columns: usize,
) -> Result<(), TensorError> {
    let last = index.len() - 1;
    for (row, chunk) in output.chunks_exact_mut(columns).enumerate() {
        index[0] = row;
        for (col, element) in chunk.iter_mut().enumerate() {
            index[last] = col;
            *element = *view.get(&*index)?;
        }
    }
    Ok(())
}

fn verify(output: &[f64], columns: usize) {
    for (row, chunk) in output.chunks_exact(columns).enumerate() {
        for (col, actual) in chunk.iter().enumerate() {
            let original = row * columns + (columns - 1 - col);
            assert_eq!(actual.to_bits(), value(original).to_bits());
        }
    }
}

fn measure(rank: usize, rows: usize, columns: usize) -> Result<(), TensorError> {
    let mut dims = vec![1; rank];
    dims[0] = rows;
    dims[rank - 1] = columns;
    let len = rows * columns;
    let input = Tensor::from_vec(&dims, (0..len).map(value).collect())?;
    let reversed = input.view().flip(rank - 1)?;
    let mut output = Tensor::from_vec(&dims, vec![f64::NAN; len])?;
    let mut index = vec![0; rank];
    materialize(&reversed, &mut index, output.as_mut_slice(), columns)?;
    verify(output.as_slice(), columns);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        materialize(
            black_box(&reversed),
            &mut index,
            black_box(output.as_mut_slice()),
            columns,
        )?;
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(output.as_slice(), columns);
    samples.sort_unstable();
    println!(
        "implementation=general_host operation=materialize_reverse dtype=f64 elements={len} rank={rank} rows={rows} last_extent={columns} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    for rank in [2, 8, 16] {
        measure(rank, 16, 257)?;
    }
    for rows in [1, 128] {
        measure(2, rows, 257)?;
    }
    Ok(())
}
