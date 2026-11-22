// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use core::ptr;

use super::{flip_axis_f32, gemm_axpy_f32, gemm_dot_f32, CpuCommand};

fn command(
    input: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    geometry: [usize; 3],
    entry: unsafe fn(&CpuCommand),
) -> CpuCommand {
    let [m, k, n] = geometry;
    CpuCommand {
        a: input.as_ptr(),
        b: rhs.as_ptr(),
        out: output.as_mut_ptr(),
        len: output.len(),
        m,
        k,
        n,
        transpose_lhs: false,
        transpose_rhs: false,
        lhs_vector: false,
        rhs_vector: false,
        scalar0: 0.0,
        scalar1: 0.0,
        arbitrary_axes: None,
        sort_scratch: ptr::null_mut(),
        entry,
    }
}

#[test]
fn dot_gemm_uses_transposed_storage_coordinates_on_both_inputs() {
    let lhs = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let rhs = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
    let mut output = [f32::NAN; 4];
    let mut cmd = command(&lhs, &rhs, &mut output, [2, 3, 2], gemm_dot_f32);
    cmd.transpose_lhs = true;
    cmd.transpose_rhs = true;
    // SAFETY: physical shapes are 3x2 and 2x3, logical shapes are 2x3 and
    // 3x2, and the separate 2x2 output is writable for the synchronous call.
    unsafe { (cmd.entry)(&cmd) };
    assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn both_scalar_gemm_entries_initialize_empty_contractions_to_positive_zero() {
    let entries: [unsafe fn(&CpuCommand); 2] = [gemm_dot_f32, gemm_axpy_f32];
    for entry in entries {
        let mut output = [f32::NAN; 6];
        let cmd = command(&[], &[], &mut output, [2, 0, 3], entry);
        // SAFETY: K=0 gives empty inputs, so neither entry reads from them.
        // The disjoint output owns exactly the six initialized writable slots.
        unsafe { (cmd.entry)(&cmd) };
        assert_eq!(output.map(f32::to_bits), [0; 6]);
    }
}

#[test]
fn flip_reorders_chunks_without_changing_nan_payloads_or_signed_zero() {
    let bits = [
        0,
        0x8000_0000,
        0x7fc0_0001,
        0x7fc0_0002,
        0x3f80_0000,
        0xbf80_0000,
        0x7fc0_0003,
        0xffc0_0004,
        0x4000_0000,
        0xc000_0000,
        0x0080_0000,
        0x8080_0000,
    ];
    let input = bits.map(f32::from_bits);
    let mut output = [0.0; 12];
    let cmd = command(&input, &[], &mut output, [2, 3, 2], flip_axis_f32);
    // SAFETY: 2*3*2 exactly covers both live, disjoint arrays. The kernel
    // reverses the three chunks within each slab and ignores the RHS.
    unsafe { (cmd.entry)(&cmd) };
    let expected = [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7].map(|source| bits[source]);
    assert_eq!(output.map(f32::to_bits), expected);
}
