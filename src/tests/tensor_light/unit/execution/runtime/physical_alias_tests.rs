// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::{external_buffers_proven_disjoint, BufferHandle, DeviceBufferHandle, HostBufferHandle};

fn host_descriptor(slice: &[u8]) -> HostBufferHandle {
    HostBufferHandle {
        address: slice.as_ptr().expose_provenance(),
        bytes: slice.len(),
    }
}

#[test]
fn host_address_proof_distinguishes_adjacency_from_one_byte_of_overlap() {
    let storage = [0_u8; 16];
    let left = BufferHandle::Host(host_descriptor(&storage[..8]));
    let adjacent = BufferHandle::Host(host_descriptor(&storage[8..]));
    let overlapping = BufferHandle::Host(host_descriptor(&storage[7..]));
    assert!(external_buffers_proven_disjoint(left, adjacent));
    assert!(external_buffers_proven_disjoint(adjacent, left));
    assert!(!external_buffers_proven_disjoint(left, overlapping));
    assert!(!external_buffers_proven_disjoint(overlapping, left));
}

#[test]
fn different_virtual_addresses_are_insufficient_for_opaque_and_device_handles() {
    // Integer metadata for the proof predicate only. These handles are not
    // bound, dereferenced or treated as proof of live device allocations.
    let host = BufferHandle::Host(HostBufferHandle {
        address: 64,
        bytes: 16,
    });
    let device = BufferHandle::Device(DeviceBufferHandle {
        device_ordinal: 1,
        address: 256,
        bytes: 16,
    });
    let peer = BufferHandle::Device(DeviceBufferHandle {
        device_ordinal: 2,
        address: 512,
        bytes: 16,
    });
    let alternate = [
        BufferHandle::HostMappedZeroCopy {
            host: HostBufferHandle {
                address: 128,
                bytes: 16,
            },
            device_address: 512,
            device_ordinal: 1,
        },
        BufferHandle::Unified {
            host_address: 192,
            device_address: 768,
            bytes: 16,
            device_ordinal: 1,
        },
        BufferHandle::External {
            host_address: Some(512),
            device_address: None,
            bytes: 16,
            device_ordinal: None,
        },
        device,
    ];
    for handle in alternate {
        assert!(!external_buffers_proven_disjoint(host, handle));
        assert!(!external_buffers_proven_disjoint(handle, host));
    }
    assert!(!external_buffers_proven_disjoint(device, peer));
    let same_device = BufferHandle::Device(DeviceBufferHandle {
        device_ordinal: 1,
        address: 512,
        bytes: 16,
    });
    assert!(!external_buffers_proven_disjoint(device, same_device));
}

#[test]
fn empty_buffers_have_no_writable_intersection_even_with_opaque_handles() {
    let nonempty = BufferHandle::Device(DeviceBufferHandle {
        device_ordinal: 1,
        address: 32,
        bytes: 4,
    });
    let empty = BufferHandle::External {
        host_address: None,
        device_address: None,
        bytes: 0,
        device_ordinal: None,
    };
    assert!(external_buffers_proven_disjoint(nonempty, empty));
    assert!(external_buffers_proven_disjoint(empty, nonempty));
}
