// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use crate::tensor_light::aot::cpu_primary_worker_seed;
use crate::tensor_light::execution::{
    CpuArchitecture, CpuHardwareContract, CpuLogicalProcessor, FeatureMask,
};

const fn processor(group: u16, logical: u16, core: u16) -> CpuLogicalProcessor {
    CpuLogicalProcessor {
        processor_group: group,
        logical_cpu_id: logical,
        physical_core_id: core,
        numa_node: group,
        package_id: group,
        efficiency_class: 0,
    }
}

const PROCESSORS: [CpuLogicalProcessor; 4] = [
    processor(0, 0, 0),
    processor(0, 1, 0),
    processor(1, 0, 1),
    processor(1, 1, 1),
];

const HARDWARE: CpuHardwareContract = CpuHardwareContract {
    architecture: CpuArchitecture::X86_64,
    features: FeatureMask(0),
    logical_cpu_count: 4,
    physical_core_count: 2,
    processors: &PROCESSORS,
    numa_node_count: 2,
    cache_line_bytes: 64,
    l1_data_bytes_per_core: 32 * 1024,
    l2_bytes_per_core: 256 * 1024,
    l3_bytes_shared: 4 * 1024 * 1024,
    max_vector_bytes: 16,
    sve_vector_bytes: 0,
    sme_vector_bytes: 0,
    rvv_vector_bytes: 0,
};

#[test]
fn worker_count_distinguishes_physical_cores_and_smt_siblings() {
    // Logical IDs can repeat across processor groups.
    assert_eq!(cpu_primary_worker_seed(&HARDWARE, true), 2);
    assert_eq!(cpu_primary_worker_seed(&HARDWARE, false), 4);
}

#[test]
fn incomplete_or_inconsistent_inventories_disable_both_worker_seeds() {
    const DUPLICATE_LOGICAL: [CpuLogicalProcessor; 4] =
        [PROCESSORS[0], PROCESSORS[0], PROCESSORS[2], PROCESSORS[3]];
    const UNREPRESENTED_CORE: [CpuLogicalProcessor; 4] = [
        PROCESSORS[0],
        PROCESSORS[1],
        processor(1, 0, 0),
        processor(1, 1, 0),
    ];
    const OUT_OF_RANGE_CORE: [CpuLogicalProcessor; 4] = [
        PROCESSORS[0],
        PROCESSORS[1],
        PROCESSORS[2],
        processor(1, 1, 2),
    ];

    for hardware in [
        CpuHardwareContract {
            processors: &[],
            ..HARDWARE
        },
        CpuHardwareContract {
            processors: &DUPLICATE_LOGICAL,
            ..HARDWARE
        },
        CpuHardwareContract {
            processors: &UNREPRESENTED_CORE,
            ..HARDWARE
        },
        CpuHardwareContract {
            processors: &OUT_OF_RANGE_CORE,
            ..HARDWARE
        },
        CpuHardwareContract {
            numa_node_count: 1,
            ..HARDWARE
        },
    ] {
        assert_eq!(cpu_primary_worker_seed(&hardware, true), 0, "{hardware:?}");
        assert_eq!(cpu_primary_worker_seed(&hardware, false), 0, "{hardware:?}");
    }
}
