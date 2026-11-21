// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

//! Measures actual Light canonical workload validation for CPU-only graphs.
//! Fixtures are static graph descriptors. No frozen plan evidence or native
//! device admission is constructed. Graph setup is outside the timed boundary.

use momiru::tensor_light::execution::{
    canonical_elementwise_broadcast_signature, AccuracyContract, CanonicalNode,
    CanonicalStaticDataflow, CanonicalWorkload, CpuArchitecture, CpuHardwareContract,
    CpuLogicalProcessor, ElementwiseKind, ExecutionTopologyContract, FeatureMask, HardwareContract,
    LayoutKind, MemoryTopologyContract, NumericalTolerance, Operation, PlanError, QuantizationKind,
    QuantizationSpec, ReusePattern, ScaleGranularity, SparsityPattern, SparsitySpec, StaticShape,
    TensorLayout, TensorRole, TensorSpec,
};
use momiru::tensor_light::DType;
use momiru::tensor_light::MAX_RANK;
use std::{hint::black_box, time::Instant};

fn tensor(id: u32) -> TensorSpec {
    let mut strides = [0; MAX_RANK];
    strides[..2].copy_from_slice(&[64, 1]);
    TensorSpec {
        id,
        shape: StaticShape::Rank2([32, 64]),
        dtype: DType::F32,
        role: TensorRole::Data,
        layout: TensorLayout {
            kind: LayoutKind::RowMajorContiguous,
            strides,
            block: [0; MAX_RANK],
            alignment_bytes: 4,
        },
        accuracy: AccuracyContract::StrictFp32,
        tolerance: NumericalTolerance {
            max_abs_error_f32_bits: 0,
            max_rel_error_f32_bits: 0,
        },
        reuse: ReusePattern::SingleUse,
        sparsity: SparsitySpec {
            pattern: SparsityPattern::Dense,
            block_m: 0,
            block_n: 0,
            metadata_tensor: None,
        },
        quantization: QuantizationSpec {
            kind: QuantizationKind::None,
            granularity: ScaleGranularity::None,
            axis0: 0,
            axis1: 0,
            block0: 0,
            block1: 0,
            scale_tensor: None,
            secondary_scale_tensor: None,
            zero_point_tensor: None,
        },
    }
}

fn workload(node_count: u32) -> Result<CanonicalWorkload, PlanError> {
    let io_shape = StaticShape::Rank2([32, 64]);
    let tensors = (0..=node_count).map(tensor).collect::<Vec<_>>();
    let signature = canonical_elementwise_broadcast_signature(&[io_shape], io_shape)?;
    let nodes = (0..node_count)
        .map(|id| CanonicalNode {
            id,
            operation: Operation::Elementwise {
                kind: ElementwiseKind::Negate,
                arity: 1,
                broadcast_signature: signature,
            },
            dependencies: if id == 0 {
                &[]
            } else {
                Box::leak(Box::new([id - 1]))
            },
            inputs: Box::leak(Box::new([id])),
            outputs: Box::leak(Box::new([id + 1])),
        })
        .collect::<Vec<_>>();
    let cpu = CpuHardwareContract {
        architecture: CpuArchitecture::Other,
        features: FeatureMask(0),
        logical_cpu_count: 1,
        physical_core_count: 1,
        numa_node_count: 1,
        processors: &[CpuLogicalProcessor {
            processor_group: 0,
            logical_cpu_id: 0,
            physical_core_id: 0,
            numa_node: 0,
            package_id: 0,
            efficiency_class: 0,
        }],
        cache_line_bytes: 64,
        l1_data_bytes_per_core: 0,
        l2_bytes_per_core: 0,
        l3_bytes_shared: 0,
        max_vector_bytes: 0,
        sve_vector_bytes: 0,
        sme_vector_bytes: 0,
        rvv_vector_bytes: 0,
    };
    // The public graph API requires static descriptors. The two small
    // measurement fixtures and their negative controls live for this process.
    Ok(CanonicalWorkload {
        graph: CanonicalStaticDataflow {
            tensors: Box::leak(tensors.into_boxed_slice()),
            nodes: Box::leak(nodes.into_boxed_slice()),
        },
        hardware: HardwareContract::canonical(Some(cpu), &[]),
        memory_topology: MemoryTopologyContract {
            host_numa_nodes: &[0],
            host_numa_distances: &[],
            gpu_device_ordinals: &[],
            peer_access: &[],
            supports_pinned_host_memory: false,
            supports_mapped_host_memory: false,
            supports_unified_memory: false,
        },
        execution_topology: ExecutionTopologyContract {
            cpu_enabled: true,
            gpu_device_ordinals: &[],
            allow_cpu_gpu_partitioned: false,
            allow_cpu_gpu_pipeline: false,
        },
    })
}

fn measure(count: u32) -> Result<(), PlanError> {
    let valid = workload(count)?;
    valid.validate()?;
    let mut wrong_nodes = valid.graph.nodes.to_vec();
    wrong_nodes.last_mut().expect("nonempty chain").dependencies = &[];
    let mut invalid = valid;
    invalid.graph.nodes = Box::leak(wrong_nodes.into_boxed_slice());
    assert_eq!(invalid.validate(), Err(PlanError::MissingDataDependency));

    let mut samples = [0; 9];
    for sample in &mut samples {
        let start = Instant::now();
        let result = black_box(black_box(&valid).validate());
        *sample = start.elapsed().as_nanos();
        result?;
    }
    valid.validate()?;
    samples.sort_unstable();
    println!(
        "implementation=light_canonical_validator operation=graph_and_topology_validation nodes={count} iterations={} median_ns={}",
        samples.len(), samples[samples.len() / 2]
    );
    Ok(())
}

fn main() -> Result<(), PlanError> {
    measure(8)?;
    measure(64)
}
