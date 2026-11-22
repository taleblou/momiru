// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{
    buffer_address_ranges_representable, byte_ranges_overlap, device_byte_range_representable,
    host_byte_range_representable, BufferHandle, DeviceBufferHandle, HostBufferHandle,
};

// These addresses are integer descriptors only. No test forms a pointer from
// the values, claims ownership of the described storage, or launches a kernel.
#[test]
fn host_rejects_null_nonempty_ranges_and_limits_exclusive_end_and_slice_size() {
    assert!(host_byte_range_representable(0, 0));
    assert!(host_byte_range_representable(usize::MAX, 0));
    assert!(!host_byte_range_representable(0, 1));
    assert!(host_byte_range_representable(usize::MAX - 1, 1));
    assert!(!host_byte_range_representable(usize::MAX - 1, 2));
    assert!(host_byte_range_representable(1, isize::MAX as usize));
    assert!(!host_byte_range_representable(1, isize::MAX as usize + 1));
}

#[test]
fn device_uses_u64_end_checks_independent_of_the_host_address_width() {
    assert!(device_byte_range_representable(0, 0));
    assert!(!device_byte_range_representable(0, 1));
    assert!(device_byte_range_representable(u64::MAX - 1, 1));
    assert!(!device_byte_range_representable(u64::MAX - 1, 2));
    assert!(!buffer_address_ranges_representable(BufferHandle::Device(
        DeviceBufferHandle {
            device_ordinal: 0,
            address: u64::MAX,
            bytes: 1
        }
    )));
}

#[test]
fn dual_domain_descriptors_require_each_present_range_to_fit() {
    let valid_host = HostBufferHandle {
        address: usize::MAX - 4,
        bytes: 4,
    };
    let invalid_host = HostBufferHandle {
        address: usize::MAX - 3,
        bytes: 4,
    };
    let valid_device = u64::MAX - 4;
    let invalid_device = u64::MAX - 3;
    assert!(buffer_address_ranges_representable(
        BufferHandle::HostPinned(valid_host)
    ));
    assert!(!buffer_address_ranges_representable(BufferHandle::Host(
        invalid_host
    )));
    for (host, device_address, expected) in [
        (valid_host, valid_device, true),
        (invalid_host, valid_device, false),
        (valid_host, invalid_device, false),
    ] {
        assert_eq!(
            buffer_address_ranges_representable(BufferHandle::HostMappedZeroCopy {
                host,
                device_address,
                device_ordinal: 0,
            }),
            expected
        );
        assert_eq!(
            buffer_address_ranges_representable(BufferHandle::Unified {
                host_address: host.address,
                device_address,
                bytes: host.bytes,
                device_ordinal: 0,
            }),
            expected
        );
        assert_eq!(
            buffer_address_ranges_representable(BufferHandle::External {
                host_address: Some(host.address),
                device_address: Some(device_address),
                bytes: host.bytes,
                device_ordinal: Some(0),
            }),
            expected
        );
    }
    assert!(buffer_address_ranges_representable(
        BufferHandle::External {
            host_address: None,
            device_address: Some(valid_device),
            bytes: 4,
            device_ordinal: Some(0),
        }
    ));
}

#[test]
fn physical_overlap_excludes_empty_and_touching_ranges() {
    let start = u128::from(u64::MAX) - 8;
    assert!(!byte_ranges_overlap(start, 4, start + 4, 4));
    assert!(byte_ranges_overlap(start, 5, start + 4, 4));
    assert!(byte_ranges_overlap(start + 4, 4, start, 5));
    assert!(!byte_ranges_overlap(start, 8, start + 1, 0));
}
