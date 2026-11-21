// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! General checked reads over dense storage and a gapped negative-stride view.
//! Coordinates and views are prepared up front. The timed loop includes the
//! checked read and checksum accumulation.

use momiru::tensor::{Tensor, TensorError, TensorView};
use std::{hint::black_box, time::Instant};

const ROWS: usize = 96;
const COLS: usize = 257;
const QUERIES: usize = 8_193;
const VIEW_COLS: usize = COLS.div_ceil(2);

fn value(offset: usize) -> u64 {
    (1 << 55) + (offset as u64 * 31) % 4_091
}

fn measure(
    operation: &str,
    queries: &[[usize; 2]],
    read: impl Fn([usize; 2]) -> Result<u64, TensorError>,
    column: impl Fn(usize) -> usize,
) -> Result<(), TensorError> {
    let mut expected_sum = 0u64;
    for &[row, col] in queries {
        let expected = value(row * COLS + column(col));
        assert_eq!(read([row, col])?, expected);
        expected_sum = expected_sum.wrapping_add(expected);
    }

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        let mut checksum = 0u64;
        for &query in black_box(queries) {
            checksum = checksum.wrapping_add(read(query)?);
        }
        let checksum = black_box(checksum);
        *sample = start.elapsed().as_nanos();
        assert_eq!(checksum, expected_sum);
    }
    for &[row, col] in queries {
        assert_eq!(read([row, col])?, value(row * COLS + column(col)));
    }
    samples.sort_unstable();
    println!(
        "implementation=general_host operation={operation} dtype=u64 shape={ROWS}x{COLS} queries={} iterations={} median_ns={}",
        queries.len(), samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::from_vec([ROWS, COLS], (0..ROWS * COLS).map(value).collect())?;
    let reversed = TensorView::new(
        [ROWS, VIEW_COLS],
        [COLS as isize, -2],
        (COLS - 1) as isize,
        tensor.as_slice(),
    )?;
    let queries: Vec<_> = (0..QUERIES)
        .map(|i| [(i * 17 + 7) % ROWS, (i * 41 + 3) % VIEW_COLS])
        .collect();

    // Both paths read the same backing elements in the same order.
    let physical_column = |col| COLS - 1 - 2 * col;
    let dense_queries: Vec<_> = queries
        .iter()
        .map(|&[row, col]| [row, physical_column(col)])
        .collect();
    measure(
        "dense_checked_read",
        &dense_queries,
        |index| Ok(*black_box(&tensor).get(index)?),
        |col| col,
    )?;
    measure(
        "gapped_negative_stride_read",
        &queries,
        |index| Ok(*black_box(&reversed).get(index)?),
        physical_column,
    )
}
