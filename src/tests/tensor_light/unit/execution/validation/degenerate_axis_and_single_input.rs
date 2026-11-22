// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use crate::tensor_light::execution::{
    AccuracyContract, AxisList, CanonicalNode, CanonicalStaticDataflow, CanonicalWorkload,
    CpuArchitecture, CpuHardwareContract, CpuLogicalProcessor, ExecutionTopologyContract,
    FeatureMask, HardwareContract, LayoutKind, MemoryTopologyContract, NumericalTolerance,
    Operation, PlanError, QuantizationKind, QuantizationSpec, ReusePattern, ScaleGranularity,
    SparsityPattern, SparsitySpec, StaticShape, TensorId, TensorLayout, TensorRole, TensorSpec,
};
use crate::tensor_light::{DType, MAX_RANK};

pub(super) fn tensor(
    id: TensorId,
    shape: StaticShape,
    dtype: DType,
    role: TensorRole,
) -> TensorSpec {
    let mut strides = [0; MAX_RANK];
    if shape.element_count() != Some(0) {
        let mut stride = 1;
        for axis in (0..shape.rank()).rev() {
            strides[axis] = stride;
            stride *= isize::try_from(shape.dim(axis).unwrap()).unwrap();
        }
    }
    TensorSpec {
        id,
        shape,
        dtype,
        role,
        layout: TensorLayout {
            kind: LayoutKind::RowMajorContiguous,
            strides,
            block: [0; MAX_RANK],
            alignment_bytes: dtype.size_bytes() as u32,
        },
        accuracy: if dtype == DType::F32 {
            AccuracyContract::StrictFp32
        } else {
            AccuracyContract::Exact
        },
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

pub(super) fn value(id: TensorId, shape: StaticShape) -> TensorSpec {
    tensor(id, shape, DType::F32, TensorRole::Data)
}

pub(super) fn validate_operation(
    operation: Operation,
    tensors: Vec<TensorSpec>,
    inputs: &'static [TensorId],
    output: TensorId,
) -> Result<(), PlanError> {
    // Static descriptors are required by the canonical graph API. These
    // bounded test allocations live for the suite and carry no plan evidence.
    let tensors = Box::leak(tensors.into_boxed_slice());
    let outputs = Box::leak(Box::new([output]));
    let nodes = Box::leak(Box::new([CanonicalNode {
        id: 0,
        operation,
        dependencies: &[],
        inputs,
        outputs,
    }]));
    let cpu = CpuHardwareContract {
        architecture: CpuArchitecture::X86_64,
        features: FeatureMask(0),
        logical_cpu_count: 1,
        physical_core_count: 1,
        processors: &[CpuLogicalProcessor {
            processor_group: 0,
            logical_cpu_id: 0,
            physical_core_id: 0,
            numa_node: 0,
            package_id: 0,
            efficiency_class: 0,
        }],
        numa_node_count: 1,
        cache_line_bytes: 64,
        l1_data_bytes_per_core: 0,
        l2_bytes_per_core: 0,
        l3_bytes_shared: 0,
        max_vector_bytes: 0,
        sve_vector_bytes: 0,
        sme_vector_bytes: 0,
        rvv_vector_bytes: 0,
    };
    CanonicalWorkload {
        graph: CanonicalStaticDataflow { tensors, nodes },
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
    }
    .validate()
}

fn unary(operation: Operation, input: StaticShape, output: StaticShape) -> Result<(), PlanError> {
    validate_operation(operation, vec![value(0, input), value(1, output)], &[0], 1)
}

#[test]
fn flip_accepts_singleton_and_zero_axes_but_requires_the_exact_shape() {
    let mut axes = AxisList::empty();
    axes.len = 1;
    axes.axes[0] = 1;
    let op = Operation::Flip { axes };
    for extent in [0, 1] {
        let shape = StaticShape::Rank3([2, extent, 4]);
        assert_eq!(unary(op, shape, shape), Ok(()));
        assert_eq!(
            unary(op, shape, StaticShape::Rank3([4, extent, 2])),
            Err(PlanError::InvalidOperationSpec)
        );
    }
    axes.axes[0] = 2;
    let shape = StaticShape::Rank2([2, 1]);
    assert_eq!(
        unary(Operation::Flip { axes }, shape, shape),
        Err(PlanError::InvalidOperationSpec)
    );
}

#[test]
fn single_input_concat_is_identity_but_stack_inserts_a_singleton() {
    let input = StaticShape::Rank2([0, 3]);
    assert_eq!(unary(Operation::Concat { axis: 1 }, input, input), Ok(()));
    assert_eq!(
        unary(
            Operation::Concat { axis: 1 },
            input,
            StaticShape::Rank2([0, 4])
        ),
        Err(PlanError::InvalidOperationSpec)
    );
    let stack = Operation::Stack { axis: 1 };
    assert_eq!(unary(stack, input, StaticShape::Rank3([0, 1, 3])), Ok(()));
    assert_eq!(
        unary(stack, input, input),
        Err(PlanError::InvalidOperationSpec)
    );
    assert_eq!(
        unary(stack, input, StaticShape::Rank3([0, 0, 3])),
        Err(PlanError::InvalidOperationSpec)
    );
    assert_eq!(
        unary(
            Operation::Stack { axis: 0 },
            StaticShape::Rank0,
            StaticShape::Rank1([1])
        ),
        Ok(())
    );
}
