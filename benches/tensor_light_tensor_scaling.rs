// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-License-Identifier: MIT

//! Measures Light fixed-rank logical traversal of a prebuilt reverse view.
//! Output is preallocated. Report element count, rank and last-axis extent
//! for each case so that distinct layouts remain visible.

use momiru::tensor_light::{Rank, SignedTensorView, Tensor, TensorError, ValidRank};
use std::{hint::black_box, time::Instant};

fn value(index: usize) -> f32 {
    (index % 4093) as f32 / 16.0
}

fn materialize<const R: usize>(
    view: &SignedTensorView<'_, R>,
    output: &mut [f32],
) -> Result<(), TensorError>
where
    Rank<R>: ValidRank,
{
    let mut index = [0; R];
    for result in output {
        *result = view.get(index)?;
        for axis in (0..R).rev() {
            index[axis] += 1;
            if index[axis] < view.dims()[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

fn verify(output: &[f32], last: usize) {
    for (logical, &actual) in output.iter().enumerate() {
        let source = (logical / last) * last + last - 1 - logical % last;
        assert_eq!(actual, value(source), "logical offset {logical}");
    }
}

fn measure<const R: usize>(dims: [usize; R]) -> Result<(), TensorError>
where
    Rank<R>: ValidRank,
{
    let len: usize = dims.iter().product();
    let last = dims[R - 1];
    let input = Tensor::<R>::try_from_vec(dims, (0..len).map(value).collect())?;
    let reverse = input.signed_view()?.flip(R - 1)?;
    let mut output = Tensor::<R>::zeros(dims)?;
    materialize(&reverse, output.as_mut_slice())?;
    verify(output.as_slice(), last);

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        materialize(black_box(&reverse), black_box(output.as_mut_slice()))?;
        black_box(output.as_slice());
        *sample = start.elapsed().as_nanos();
    }
    verify(output.as_slice(), last);
    samples.sort_unstable();
    println!(
        "implementation=light_host operation=materialize_reverse dtype=f32 elements={len} rank={R} last_extent={last} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), TensorError> {
    measure([16, 16])?;
    measure([2, 2, 2, 2, 1, 1, 1, 16])?;
    measure([1, 257])?;
    measure([16, 257])?;
    measure([128, 257])
}
