// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | EXECUTION ENGINE
// ==============================================================================
//
// Project:       Momiru
// Component:     Execution
// Subcomponent:  Validation
// Module:        momiru::tensor_light::execution::validation
// Source:        src/tensor_light/execution/validation.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Semantic and resource admission validation for immutable UASDEA execution
//   graphs and plans.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Reject malformed or unsupported graph and workload contracts.
//   - Check dimensional, type, layout and execution compatibility.
//   - Keep validation separate from numeric kernel execution.
//
// ------------------------------------------------------------------------------
// DESIGN AND ENGINEERING GUIDELINES
// ------------------------------------------------------------------------------
//
// Correctness:
//   - Keep the concrete tensor element type f32 and fixed ranks 0 through 8.
//   - Validate shapes, strides, broadcasting and dimensional compatibility.
//   - Preserve frozen UASDEA plan semantics and placement contracts.
//   - Distinguish host-side metadata operations from plan-bound numerical work.
//   - Maintain result rank, shape and f32 numerical semantics as documented.
//   - Handle scalar rank zero, empty extents, NaN and boundary conditions.
//   - Check dimension products, byte spans and index arithmetic for overflow.
//   - Reject unsupported operations and invalid resource bindings explicitly.
//   - Make error and failure atomicity contracts clear at public boundaries.
//   - Respect Rust ownership, borrowing and thread-safety requirements.
//
// Performance Objectives:
//   - Prioritize maximum practical runtime performance without changing
//     semantics.
//   - Avoid unnecessary heap allocation, payload copying and materialization.
//   - Favor contiguous access, locality and efficient logical traversal.
//   - Hoist invariant shape, stride and feature checks out of hot loops.
//   - Minimize per-element branching, bounds rechecks and dispatch overhead.
//   - Exploit compile-time specialization or monomorphization when beneficial.
//   - Use SIMD or parallel work only when correctness and measurements justify
//     it.
//   - Keep replay free from avoidable planning, tuning and synchronization.
//   - Prefer optimized paths with safe, equivalent reference fallbacks.
//   - Use reproducible benchmarks instead of unmeasured speed assertions.
//
// Memory Safety:
//   - Prefer safe Rust APIs that preserve zero-cost ownership and borrowing.
//   - Prohibit unintended aliasing, invalid references and data races.
//   - Restrict unsafe code to documented, reviewed and justified boundaries.
//   - Document soundness invariants for every unsafe operation or block.
//   - Validate bounds, allocation provenance, alignment and initialization.
//   - Preserve correct slice, view, stride and owner lifetime relationships.
//   - Keep device or foreign handles opaque unless validated by real providers.
//   - Require exclusive mutable access where aliasing can occur.
//   - Maintain soundness across public and internal API boundaries.
//
// Maintainability:
//   - Follow idiomatic Rust conventions and consistent module organization.
//   - Keep responsibilities explicit and components independently testable.
//   - Centralize shared shape, indexing, error and layout contracts.
//   - Keep public behavior separate from implementation optimizations.
//   - Document public APIs with Rustdoc and preserve existing module docs.
//   - Maintain public compatibility unless changes are explicitly intended.
//   - Ensure optimized and reference paths have compatible semantics.
//   - Update unit, integration and regression tests for behavioral changes.
//   - Verify performance changes with representative repeatable benchmarks.
//   - Favor measurable specialization over unnecessary duplicated complexity.
//
// ------------------------------------------------------------------------------
// PROJECT INFORMATION
// ------------------------------------------------------------------------------
//
// Authors:
//   Morteza Taleblou
//   Mitra Daneshmand
//
// Organization:  https://github.com/momiru
// Repository:    https://github.com/taleblou/momiru
//
// Maintainer Website:
//   https://taleblou.ir/
//
// Maintainer LinkedIn:
//   https://www.linkedin.com/in/morteza-taleblou
//
// ------------------------------------------------------------------------------
// LICENSING AND ATTRIBUTION
// ------------------------------------------------------------------------------
//
// License:       MIT
// License File:  LICENSE
//
// Copyright and Licensing:
//   This source file is distributed under the MIT License.
//   The SPDX declarations above identify its applicable license and
//   stated copyright holders.
//
//   Redistributions containing this software or substantial portions
//   of it must retain the applicable copyright and permission notices
//   as required by the MIT License.
//
// Attribution:
//   Acknowledgment of the Momiru project in downstream documentation
//   is appreciated but is not an additional licensing requirement.
//
// ==============================================================================

use crate::tensor_light::tensor::DType;

use super::flip_alias::flipped_alias_delta;
use super::ir::{
    canonical_broadcast_extent, canonical_elementwise_broadcast_signature, AccuracyContract, AxisSelector, AttentionMaskKind, DenseLinalgKind,
    ElementwiseKind, FourierTransformKind, KernelMathMode, LayoutKind, NodeId, NormalizationKind,
    Operation, PadMode, QuantizationKind, ReductionKind, ResampleMode, ReusePattern,
    ScalarElementwiseKind, ScaleGranularity, ScanOutput, SortOutput,
    SparsityPattern, StaticShape, TensorId,
    TensorLayout, TensorRole, TensorSpec,
};
use super::{
    ArenaId, EventId, ExecutionContextId, MemorySlotId, PlanError, StorageId, StreamId, SyncId,
    TransferBatchId, TransferId,
};
use super::plan::{
    AsyncCopyKind, BarrierKind, ClusterLaunchControlMode, ClusterSchedulingPolicy, CpuGpuMode,
    CpuDispatchMode, CpuKernelPlan, CpuKernelSource, CpuMemoryAccessPlan, CpuParallelKind,
    CpuVectorAccessKind, DeviceAllocationPolicy, EventPlan,
    ExecutionTarget, FrozenExecutionPlan, FrozenNode, GpuExecutionContextPlan, GpuKernelPlan,
    GpuKernelSource,
    GpuCacheEvictionPriority, GpuLoadCachePolicy, GpuPartitionKind, GpuStoreCachePolicy,
    GraphReplayMode, HostBoundaryClass, HostMemoryPolicy, KernelAssumptionPlan,
    KernelFamily, L2AccessProperty, LifetimeClass, MatrixAcceleration, MemoryArenaPlan, MemoryClass,
    MemcpySrcAccessOrder, MemoryLocation, MemorySlotPlan, NumaPolicy, PagePolicy, PgoMode, Placement,
    PortableClusterMode, PostLinkOptimization, PrefetchKind, ResidencyPolicy, RustFloatMode,
    RustPanicStrategy, ScheduleItem, SharedMemoryMode, SmPartitionKind, StorageAccess, StorePolicy,
    StreamPlan,
    SyncKind, TargetIoPlan, TensorMapDirection, TensorMapInterleave, TensorMapOobFill,
    TensorMapKind, TensorMapStoreReduction, TensorMapSwizzle, TensorStoragePlan,
    TransferBatchKind, TransferBatchPlan, TransferKind, TransferPlan,
};
use super::target::{
    cpu_feature, gpu_feature, ArtifactKind, ArtifactRef, CpuLogicalProcessor, FeatureMask,
    GpuHardwareContract, LibraryAlgorithmPlan, LibraryProvider,
};

impl FrozenExecutionPlan {
    /// Performs binding-time structural, semantic and capability validation.
    /// Deterministic replay must not repeat these checks.
    pub fn validate(&self) -> Result<(), PlanError> {
        // Strict UASDEA never instantiates/optimizes a graph schedule online.
        // Legacy GPU graph segments require a separate pre-existing native
        // executable proof not present in this implementation: reject them.
        if !self.gpu_graph_segments.is_empty() {
            return Err(PlanError::InvalidGpuGraphSegment);
        }
        self.validate_static_payload()?;
        self.validate_identity()?;
        Ok(())
    }

    /// Validates the immutable execution payload independently of winner evidence.
    /// The offline pipeline uses this before capability gating and before costing/benchmarking
    /// so malformed candidates never enter capability or performance selection.
    pub(crate) fn validate_static_payload(&self) -> Result<(), PlanError> {
        if !self.gpu_graph_segments.is_empty() {
            return Err(PlanError::InvalidGpuGraphSegment);
        }
        self.canonical.validate()?;
        self.validate_unique_ids()?;
        self.validate_canonical_mapping()?;
        self.validate_tensor_specs()?;
        self.validate_hardware_resources()?;
        self.validate_schedule()?;
        self.validate_gpu_graph_segments()?;
        self.validate_memory()?;
        self.validate_nodes()?;
        self.validate_transfers()?;
        self.validate_transfer_batches()?;
        self.validate_sync()?;
        Ok(())
    }
}

mod semantic {
    use super::*;

// =============================================================================
// Common validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_identity(&self) -> Result<(), PlanError> {
        let seal = self.seal;
        if self.name.is_empty()
            || !self.workload.is_complete()
            || (self.hardware.profile_hash_hi == 0 && self.hardware.profile_hash_lo == 0)
            || (seal.plan_hash_hi == 0 && seal.plan_hash_lo == 0)
            || (seal.calibration_hash_hi == 0 && seal.calibration_hash_lo == 0)
            || (seal.verification_hash_hi == 0 && seal.verification_hash_lo == 0)
        {
            return Err(PlanError::InvalidPlanSeal);
        }

        // Exact identity is recomputed from the actual canonical payload.
        let expected_workload = self.canonical.signature();
        let candidate_id = self.candidate_parts().candidate_id();
        if self.key != self.workload.plan_key()
            || self.workload != expected_workload
            || self.canonical.hardware != self.hardware
            || self.canonical.graph.tensors != self.tensors
            || self.workload.hardware_hi != self.hardware.profile_hash_hi
            || self.workload.hardware_lo != self.hardware.profile_hash_lo
            || self.evidence.workload != self.workload
            || self.evidence.final_key != self.key
            || self.evidence.winner != candidate_id
            || !self.evidence.is_self_consistent()
            || !self.evidence.winner_evidence.verification.satisfies(&self.canonical)
        {
            return Err(PlanError::InvalidPlanEvidence);
        }

        let calibration = self.evidence.calibration_hash();
        let verification = self.evidence.verification_hash();
        if seal.calibration_hash_hi != calibration.0
            || seal.calibration_hash_lo != calibration.1
            || seal.verification_hash_hi != verification.0
            || seal.verification_hash_lo != verification.1
        {
            return Err(PlanError::InvalidPlanEvidence);
        }

        // The seal is recomputed from the actual immutable plan payload, not
        // merely checked for a non-zero marker.
        let payload_hash = self.compute_payload_hash();
        if seal.plan_hash_hi != payload_hash.0 || seal.plan_hash_lo != payload_hash.1 {
            return Err(PlanError::InvalidPlanSeal);
        }
        Ok(())
    }

    #[inline(always)]
    pub(super) fn find_tensor(&self, id: TensorId) -> Option<&TensorSpec> {
        self.tensors.get(id as usize).filter(|tensor| tensor.id == id)
    }

    #[inline(always)]
    pub(super) fn tensor_producer(&self, tensor: TensorId) -> Option<NodeId> {
        let mut i = 0usize;
        while i < self.nodes.len() {
            let mut o = 0usize;
            while o < self.nodes[i].outputs.len() {
                if self.nodes[i].outputs[o] == tensor {
                    return Some(self.nodes[i].id);
                }
                o += 1;
            }
            i += 1;
        }
        None
    }

    #[inline(always)]
    pub(super) fn find_cpu_processor(
        &self,
        processor_group: u16,
        logical_cpu_id: u16,
    ) -> Option<&CpuLogicalProcessor> {
        let cpu = self.hardware.cpu.as_ref()?;
        let mut i = 0usize;
        while i < cpu.processors.len() {
            let p = &cpu.processors[i];
            if p.processor_group == processor_group && p.logical_cpu_id == logical_cpu_id {
                return Some(p);
            }
            i += 1;
        }
        None
    }

    #[inline(always)]
    pub(crate) fn find_gpu(&self, device_ordinal: u16) -> Option<&GpuHardwareContract> {
        let mut i = 0usize;
        while i < self.hardware.gpus.len() {
            if self.hardware.gpus[i].device_ordinal == device_ordinal {
                return Some(&self.hardware.gpus[i]);
            }
            i += 1;
        }
        None
    }

    #[inline(always)]
    pub(super) fn gpu_has_feature(&self, device_ordinal: u16, feature: FeatureMask) -> bool {
        self.find_gpu(device_ordinal)
            .map(|gpu| gpu.features.contains(feature))
            .unwrap_or(false)
    }

    #[inline]
    pub(super) fn all_gpus_have_feature(&self, feature: FeatureMask) -> bool {
        if self.hardware.gpus.is_empty() {
            return false;
        }
        let mut i = 0usize;
        while i < self.hardware.gpus.len() {
            if !self.hardware.gpus[i].features.contains(feature) {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline(always)]
    pub(super) fn find_arena(&self, id: ArenaId) -> Option<&MemoryArenaPlan> {
        self.arenas.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(crate) fn find_slot(&self, id: MemorySlotId) -> Option<&MemorySlotPlan> {
        self.memory_slots.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(crate) fn find_storage(&self, id: StorageId) -> Option<&TensorStoragePlan> {
        self.storages.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn find_context(&self, id: ExecutionContextId) -> Option<&GpuExecutionContextPlan> {
        self.execution_contexts.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn find_stream(&self, id: StreamId) -> Option<&StreamPlan> {
        self.streams.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn find_event(&self, id: EventId) -> Option<&EventPlan> {
        self.events.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn find_node(&self, id: NodeId) -> Option<&FrozenNode> {
        self.nodes.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn has_node(&self, id: NodeId) -> bool {
        self.find_node(id).is_some()
    }

    #[inline(always)]
    pub(super) fn find_transfer(&self, id: TransferId) -> Option<&TransferPlan> {
        self.transfers.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn has_transfer(&self, id: TransferId) -> bool {
        self.find_transfer(id).is_some()
    }

    #[inline(always)]
    pub(super) fn find_transfer_batch(&self, id: TransferBatchId) -> Option<&TransferBatchPlan> {
        self.transfer_batches.get(id as usize).filter(|item| item.id == id)
    }

    #[inline(always)]
    pub(super) fn has_transfer_batch(&self, id: TransferBatchId) -> bool {
        self.find_transfer_batch(id).is_some()
    }

    pub(super) fn batch_for_transfer(&self, id: TransferId) -> Option<TransferBatchId> {
        let mut i = 0usize;
        while i < self.transfer_batches.len() {
            let batch = &self.transfer_batches[i];
            let mut j = 0usize;
            while j < batch.transfers.len() {
                if batch.transfers[j] == id {
                    return Some(batch.id);
                }
                j += 1;
            }
            i += 1;
        }
        None
    }

    #[inline(always)]
    pub(super) fn has_sync(&self, id: SyncId) -> bool {
        self.synchronization.get(id as usize).is_some_and(|item| item.id == id)
    }

    pub(super) fn node_schedule_start_position(&self, id: NodeId) -> Option<usize> {
        let node = self.nodes.get(id as usize).filter(|node| node.id == id)?;
        let mut i = 0usize;
        while i < self.schedule.len() {
            match self.schedule[i] {
                ScheduleItem::Node(node_id) if node_id == id && node.placement != Placement::CpuGpu => {
                    return Some(i);
                }
                ScheduleItem::CpuGpuSliceLaunch { node: node_id, .. }
                    if node_id == id && node.placement == Placement::CpuGpu =>
                {
                    return Some(i);
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    pub(super) fn node_schedule_completion_position(&self, id: NodeId) -> Option<usize> {
        let node = self.nodes.get(id as usize).filter(|node| node.id == id)?;
        let mut i = 0usize;
        while i < self.schedule.len() {
            match self.schedule[i] {
                ScheduleItem::Node(node_id) if node_id == id && node.placement != Placement::CpuGpu => {
                    return Some(i);
                }
                ScheduleItem::CpuGpuSliceComplete { node: node_id, .. }
                    if node_id == id && node.placement == Placement::CpuGpu =>
                {
                    // Keep scanning: node completion is the last exact slice
                    // completion in schedule order.
                    let mut last = i;
                    let mut j = i + 1;
                    while j < self.schedule.len() {
                        if let ScheduleItem::CpuGpuSliceComplete { node: later_id, .. } = self.schedule[j] {
                            if later_id == id {
                                last = j;
                            }
                        }
                        j += 1;
                    }
                    return Some(last);
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    pub(super) fn cpu_gpu_slice_launch_position(&self, id: NodeId, slice: u16) -> Option<usize> {
        let node = self.nodes.get(id as usize).filter(|node| node.id == id)?;
        if node.placement != Placement::CpuGpu {
            return None;
        }
        let mapping = node.cpu_gpu?;
        if usize::from(slice) >= mapping.slices.len() {
            return None;
        }
        let mut i = 0usize;
        while i < self.schedule.len() {
            if self.schedule[i] == (ScheduleItem::CpuGpuSliceLaunch { node: id, slice }) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    pub(super) fn cpu_gpu_slice_completion_position(
        &self,
        id: NodeId,
        slice: u16,
    ) -> Option<usize> {
        let node = self.nodes.get(id as usize).filter(|node| node.id == id)?;
        if node.placement != Placement::CpuGpu {
            return None;
        }
        let mapping = node.cpu_gpu?;
        if usize::from(slice) >= mapping.slices.len() {
            return None;
        }
        let mut i = 0usize;
        while i < self.schedule.len() {
            if self.schedule[i] == (ScheduleItem::CpuGpuSliceComplete { node: id, slice }) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[inline]
    pub(super) fn execution_start_position(&self, node: NodeId, slice: Option<u16>) -> Option<usize> {
        match slice {
            Some(slice) => self.cpu_gpu_slice_launch_position(node, slice),
            None => self.node_schedule_start_position(node),
        }
    }

    #[inline]
    pub(super) fn execution_completion_position(
        &self,
        node: NodeId,
        slice: Option<u16>,
    ) -> Option<usize> {
        match slice {
            Some(slice) => self.cpu_gpu_slice_completion_position(node, slice),
            None => self.node_schedule_completion_position(node),
        }
    }

    pub(super) fn transfer_schedule_position(&self, id: TransferId) -> Option<usize> {
        let batch = self.batch_for_transfer(id);
        let mut i = 0usize;
        while i < self.schedule.len() {
            if self.schedule[i] == ScheduleItem::Transfer(id) {
                return Some(i);
            }
            if let Some(batch_id) = batch {
                if self.schedule[i] == ScheduleItem::TransferBatch(batch_id) {
                    return Some(i);
                }
            }
            i += 1;
        }
        None
    }

    pub(super) fn sync_schedule_position(&self, id: SyncId) -> Option<usize> {
        let mut i = 0usize;
        while i < self.schedule.len() {
            if self.schedule[i] == ScheduleItem::Sync(id) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    pub(crate) fn arena_of_storage(&self, storage: StorageId) -> Option<&MemoryArenaPlan> {
        let storage = self.find_storage(storage)?;
        let slot = self.find_slot(storage.slot)?;
        self.find_arena(slot.arena)
    }

    #[inline(always)]
    pub(super) fn arena_accessible_on_gpu(
        &self,
        arena: &MemoryArenaPlan,
        device_ordinal: u16,
    ) -> bool {
        match arena.class {
            MemoryClass::DeviceLocal => {
                if arena.device_allocation == DeviceAllocationPolicy::MulticastMapped {
                    arena.multicast_devices.contains(&device_ordinal)
                } else {
                    arena.device_ordinal == Some(device_ordinal)
                }
            }
            MemoryClass::UnifiedManaged => {
                self.gpu_has_feature(device_ordinal, gpu_feature::UNIFIED_MEMORY)
            }
            MemoryClass::HostMappedZeroCopy => {
                self.gpu_has_feature(device_ordinal, gpu_feature::MAPPED_HOST_MEMORY)
            }
            MemoryClass::HostPageableDirect => {
                arena.device_ordinal == Some(device_ordinal)
                    && self.gpu_has_feature(device_ordinal, gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS)
                    && self.gpu_has_feature(
                        device_ordinal,
                        gpu_feature::PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES,
                    )
            }
            MemoryClass::External => arena.device_ordinal == Some(device_ordinal),
            MemoryClass::HostAligned | MemoryClass::HostPinned => false,
        }
    }

}

// =============================================================================
// Contracts validation
// =============================================================================

impl FrozenExecutionPlan {
    #[inline(always)]
    pub(super) fn memory_location_order_key(location: MemoryLocation) -> (u8, u16) {
        match location {
            MemoryLocation::Host => (0, 0),
            MemoryLocation::HostNuma(node) => (1, node),
            MemoryLocation::Device(device) => (2, device),
        }
    }

    pub(super) fn validate_memory_location(
        &self,
        location: MemoryLocation,
    ) -> Result<(), PlanError> {
        match location {
            MemoryLocation::Host => Ok(()),
            MemoryLocation::HostNuma(node) => {
                let cpu = self.hardware.cpu.ok_or(PlanError::InvalidArenaPolicy)?;
                if node >= cpu.numa_node_count
                    || (!self.canonical.memory_topology.host_numa_nodes.is_empty()
                        && !self.canonical.memory_topology.host_numa_nodes.contains(&node))
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                Ok(())
            }
            MemoryLocation::Device(device) => {
                if !self.canonical.memory_topology.gpu_device_ordinals.contains(&device) {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                if !self.gpu_has_feature(device, gpu_feature::UNIFIED_MEMORY) {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
                Ok(())
            }
        }
    }

    pub(super) fn validate_hardware_resources(&self) -> Result<(), PlanError> {
        // Canonical capability payload validation is centralized on
        // HardwareContract and already runs before candidate generation.
        self.hardware.validate_canonical()?;

        let mut peer_i = 0usize;
        while peer_i < self.canonical.memory_topology.peer_access.len() {
            let pair = &self.canonical.memory_topology.peer_access[peer_i];
            if pair.src_device == pair.dst_device
                || self.find_gpu(pair.src_device).is_none()
                || self.find_gpu(pair.dst_device).is_none()
                || (pair.native_atomic_supported && pair.partial_native_atomic_supported)
                || (pair.partial_native_atomic_supported && pair.atomic_capabilities_hash == 0)
                || (!pair.partial_native_atomic_supported && pair.atomic_capabilities_hash != 0)
            {
                return Err(PlanError::InvalidMemoryTopology);
            }
            if peer_i > 0 {
                let previous = &self.canonical.memory_topology.peer_access[peer_i - 1];
                if (previous.src_device, previous.dst_device)
                    >= (pair.src_device, pair.dst_device)
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
            }
            peer_i += 1;
        }

        let mut i = 0usize;
        while i < self.execution_contexts.len() {
            let context = &self.execution_contexts[i];
            let gpu = self
                .find_gpu(context.device_ordinal)
                .ok_or(PlanError::GpuMappingOnGpuAbsentHardware)?;
            if let Some(domain) = context.sm_partition.locality_domain {
                if domain >= gpu.locality_domain_count
                    || !gpu.features.contains(gpu_feature::LOCALITY_DOMAINS)
                    || context.sm_partition.kind != SmPartitionKind::GreenContext
                {
                    return Err(PlanError::InvalidGpuExecutionContext);
                }
            }
            if context.persisting_l2_reservation_bytes > 0
                && (!gpu.features.contains(gpu_feature::PERSISTING_L2)
                    || context.persisting_l2_reservation_bytes > gpu.max_persisting_l2_bytes)
                {
                    return Err(PlanError::InvalidGpuExecutionContext);
                }
            if let Some(workqueue) = context.workqueue {
                if context.sm_partition.kind != SmPartitionKind::GreenContext
                    || !gpu.features.contains(gpu_feature::GREEN_CONTEXT_WORKQUEUE)
                    || workqueue.concurrency_limit == 0
                    || workqueue.concurrency_limit > gpu.max_workqueue_concurrency
                {
                    return Err(PlanError::InvalidGpuExecutionContext);
                }
            }

            match context.sm_partition.kind {
                SmPartitionKind::AllSms => {
                    if context.sm_partition.sm_count != 0
                        && context.sm_partition.sm_count != gpu.sm_count
                    {
                        return Err(PlanError::InvalidSmPartition);
                    }
                }
                SmPartitionKind::GreenContext => {
                    let part = &context.sm_partition;
                    if !gpu.features.contains(gpu_feature::GREEN_CONTEXT)
                        || part.sm_count < gpu.min_sm_partition_size
                        || part.sm_count > gpu.sm_count
                        || part.coscheduled_sm_count == 0
                        || !part.coscheduled_sm_count.is_multiple_of(gpu.sm_coscheduled_alignment)
                        || part.preferred_coscheduled_sm_count < part.coscheduled_sm_count
                        || !part.preferred_coscheduled_sm_count.is_multiple_of(part.coscheduled_sm_count)
                    {
                        return Err(PlanError::InvalidSmPartition);
                    }
                    if !part.allow_backfill
                        && (!part.sm_count.is_multiple_of(part.coscheduled_sm_count)
                            || !part.sm_count.is_multiple_of(gpu.sm_coscheduled_alignment))
                    {
                        return Err(PlanError::InvalidSmPartition);
                    }
                }
                SmPartitionKind::ExecutionAffinity => {
                    if !gpu.features.contains(gpu_feature::EXECUTION_AFFINITY)
                        || context.sm_partition.sm_count == 0
                        || context.sm_partition.sm_count > gpu.sm_count
                    {
                        return Err(PlanError::InvalidSmPartition);
                    }
                }
            }
            i += 1;
        }

        i = 0;
        while i < self.streams.len() {
            let stream = &self.streams[i];
            let gpu = self
                .find_gpu(stream.device_ordinal)
                .ok_or(PlanError::GpuMappingOnGpuAbsentHardware)?;
            let context = self
                .find_context(stream.execution_context)
                .ok_or(PlanError::MissingExecutionContext)?;
            if context.device_ordinal != stream.device_ordinal {
                return Err(PlanError::InvalidGpuStream);
            }
            if stream.priority != 0 && !gpu.features.contains(gpu_feature::STREAM_PRIORITIES) {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
            if stream.priority < gpu.greatest_stream_priority
                || stream.priority > gpu.least_stream_priority
            {
                return Err(PlanError::InvalidGpuStream);
            }
            i += 1;
        }

        i = 0;
        while i < self.events.len() {
            if self.find_gpu(self.events[i].device_ordinal).is_none() {
                return Err(PlanError::GpuMappingOnGpuAbsentHardware);
            }
            i += 1;
        }

        Ok(())
    }

    fn validate_dense_resource_ids<T, F>(
        items: &[T],
        duplicate_error: PlanError,
        mut id_of: F,
    ) -> Result<(), PlanError>
    where
        F: FnMut(&T) -> u32,
    {
        let mut i = 0usize;
        while i < items.len() {
            let id = id_of(&items[i]);
            let mut j = i + 1;
            while j < items.len() {
                if id == id_of(&items[j]) {
                    return Err(duplicate_error);
                }
                j += 1;
            }
            let expected = u32::try_from(i).map_err(|_| PlanError::InvalidResourceIdentity)?;
            if id != expected {
                return Err(PlanError::InvalidResourceIdentity);
            }
            i += 1;
        }
        Ok(())
    }

    fn validate_memory_group_canonicality(&self) -> Result<(), PlanError> {
        fn group_is_first(slots: &[MemorySlotPlan], index: usize, group: u32, alias: bool) -> bool {
            let mut i = 0usize;
            while i < index {
                let prior = if alias { slots[i].alias_group } else { slots[i].reuse_group };
                if prior == group {
                    return false;
                }
                i += 1;
            }
            true
        }

        fn groups_are_dense(slots: &[MemorySlotPlan], alias: bool) -> bool {
            let mut expected = 1u32;
            let mut i = 0usize;
            while i < slots.len() {
                let group = if alias { slots[i].alias_group } else { slots[i].reuse_group };
                if group != 0 && group_is_first(slots, i, group, alias) {
                    if group != expected {
                        return false;
                    }
                    expected = match expected.checked_add(1) {
                        Some(value) => value,
                        None => return false,
                    };
                }
                i += 1;
            }
            true
        }

        if !groups_are_dense(self.memory_slots, true) || !groups_are_dense(self.memory_slots, false) {
            return Err(PlanError::InvalidMemoryGrouping);
        }

        let mut i = 0usize;
        while i < self.memory_slots.len() {
            let slot = &self.memory_slots[i];
            if slot.alias_group != 0 {
                let mut meaningful = false;
                let mut j = 0usize;
                while j < self.memory_slots.len() {
                    if i != j && self.memory_slots[j].alias_group == slot.alias_group {
                        let other = &self.memory_slots[j];
                        if slot.arena == other.arena {
                            let a_end = slot.offset_bytes.checked_add(slot.bytes)
                                .ok_or(PlanError::MemorySlotOutOfBounds)?;
                            let b_end = other.offset_bytes.checked_add(other.bytes)
                                .ok_or(PlanError::MemorySlotOutOfBounds)?;
                            let bytes_overlap = slot.offset_bytes < b_end && other.offset_bytes < a_end;
                            let live_overlap = slot.live_from_step <= other.live_until_step
                                && other.live_from_step <= slot.live_until_step;
                            if bytes_overlap && live_overlap {
                                meaningful = true;
                                break;
                            }
                        }
                    }
                    j += 1;
                }
                if !meaningful {
                    return Err(PlanError::InvalidMemoryGrouping);
                }
            }

            if slot.reuse_group != 0 {
                let mut meaningful = false;
                let mut j = 0usize;
                while j < self.memory_slots.len() {
                    if i != j && self.memory_slots[j].reuse_group == slot.reuse_group {
                        let other = &self.memory_slots[j];
                        if slot.arena == other.arena {
                            let a_end = slot.offset_bytes.checked_add(slot.bytes)
                                .ok_or(PlanError::MemorySlotOutOfBounds)?;
                            let b_end = other.offset_bytes.checked_add(other.bytes)
                                .ok_or(PlanError::MemorySlotOutOfBounds)?;
                            let bytes_overlap = slot.offset_bytes < b_end && other.offset_bytes < a_end;
                            let live_overlap = slot.live_from_step <= other.live_until_step
                                && other.live_from_step <= slot.live_until_step;
                            if bytes_overlap && !live_overlap {
                                meaningful = true;
                                break;
                            }
                        }
                    }
                    j += 1;
                }
                if !meaningful {
                    return Err(PlanError::InvalidMemoryGrouping);
                }
            }
            i += 1;
        }
        Ok(())
    }

    pub(super) fn validate_unique_ids(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.tensors.len() {
            let mut j = i + 1;
            while j < self.tensors.len() {
                if self.tensors[i].id == self.tensors[j].id {
                    return Err(PlanError::DuplicateTensorId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.arenas.len() {
            let mut j = i + 1;
            while j < self.arenas.len() {
                if self.arenas[i].id == self.arenas[j].id {
                    return Err(PlanError::DuplicateArenaId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_slots.len() {
            let mut j = i + 1;
            while j < self.memory_slots.len() {
                if self.memory_slots[i].id == self.memory_slots[j].id {
                    return Err(PlanError::DuplicateMemorySlotId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_slots.len() {
            let a = &self.memory_slots[i];
            let a_end = a
                .offset_bytes
                .checked_add(a.bytes)
                .ok_or(PlanError::MemorySlotOutOfBounds)?;
            let mut j = i + 1;
            while j < self.memory_slots.len() {
                let b = &self.memory_slots[j];
                if a.arena == b.arena {
                    let b_end = b
                        .offset_bytes
                        .checked_add(b.bytes)
                        .ok_or(PlanError::MemorySlotOutOfBounds)?;
                    let byte_overlap = a.offset_bytes < b_end && b.offset_bytes < a_end;
                    let live_overlap = a.live_from_step <= b.live_until_step
                        && b.live_from_step <= a.live_until_step;
                    let intentional_alias = a.alias_group != 0 && a.alias_group == b.alias_group;
                    let intentional_reuse = a.reuse_group != 0 && a.reuse_group == b.reuse_group;
                    if byte_overlap && live_overlap && !intentional_alias {
                        return Err(PlanError::MemoryAliasConflict);
                    }
                    if byte_overlap && !live_overlap && !intentional_reuse {
                        return Err(PlanError::MemoryAliasConflict);
                    }
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.storages.len() {
            let mut j = i + 1;
            while j < self.storages.len() {
                if self.storages[i].id == self.storages[j].id {
                    return Err(PlanError::DuplicateStorageId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.execution_contexts.len() {
            let mut j = i + 1;
            while j < self.execution_contexts.len() {
                if self.execution_contexts[i].id == self.execution_contexts[j].id {
                    return Err(PlanError::DuplicateExecutionContextId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.streams.len() {
            let mut j = i + 1;
            while j < self.streams.len() {
                if self.streams[i].id == self.streams[j].id {
                    return Err(PlanError::DuplicateStreamId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.events.len() {
            let mut j = i + 1;
            while j < self.events.len() {
                if self.events[i].id == self.events[j].id {
                    return Err(PlanError::DuplicateEventId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.nodes.len() {
            let mut j = i + 1;
            while j < self.nodes.len() {
                if self.nodes[i].id == self.nodes[j].id {
                    return Err(PlanError::DuplicateNodeId);
                }
                j += 1;
            }
            i += 1;
        }

        // Every logical tensor has at most one producer node. In-place execution
        // uses distinct logical tensor IDs; aliasing belongs to the storage plan.
        i = 0;
        while i < self.nodes.len() {
            let mut oi = 0usize;
            while oi < self.nodes[i].outputs.len() {
                let tensor = self.nodes[i].outputs[oi];
                let mut oj = oi + 1;
                while oj < self.nodes[i].outputs.len() {
                    if self.nodes[i].outputs[oj] == tensor {
                        return Err(PlanError::DuplicateTensorProducer);
                    }
                    oj += 1;
                }
                let mut j = i + 1;
                while j < self.nodes.len() {
                    let mut k = 0usize;
                    while k < self.nodes[j].outputs.len() {
                        if self.nodes[j].outputs[k] == tensor {
                            return Err(PlanError::DuplicateTensorProducer);
                        }
                        k += 1;
                    }
                    j += 1;
                }
                oi += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.transfers.len() {
            let mut j = i + 1;
            while j < self.transfers.len() {
                if self.transfers[i].id == self.transfers[j].id {
                    return Err(PlanError::DuplicateTransferId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.transfer_batches.len() {
            let mut j = i + 1;
            while j < self.transfer_batches.len() {
                if self.transfer_batches[i].id == self.transfer_batches[j].id {
                    return Err(PlanError::DuplicateTransferBatchId);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.synchronization.len() {
            let mut j = i + 1;
            while j < self.synchronization.len() {
                if self.synchronization[i].id == self.synchronization[j].id {
                    return Err(PlanError::DuplicateSyncId);
                }
                j += 1;
            }
            i += 1;
        }

        // Execution-resource IDs are canonical array indices. Besides making
        // the frozen representation unique under renumbering, this allows
        // binding-time resolution to use direct O(1) indexing.
        Self::validate_dense_resource_ids(self.arenas, PlanError::DuplicateArenaId, |x| x.id)?;
        Self::validate_dense_resource_ids(self.memory_slots, PlanError::DuplicateMemorySlotId, |x| x.id)?;
        Self::validate_dense_resource_ids(self.storages, PlanError::DuplicateStorageId, |x| x.id)?;
        Self::validate_dense_resource_ids(
            self.execution_contexts,
            PlanError::DuplicateExecutionContextId,
            |x| x.id,
        )?;
        Self::validate_dense_resource_ids(self.streams, PlanError::DuplicateStreamId, |x| x.id)?;
        Self::validate_dense_resource_ids(self.events, PlanError::DuplicateEventId, |x| x.id)?;
        Self::validate_dense_resource_ids(self.nodes, PlanError::DuplicateNodeId, |x| x.id)?;
        Self::validate_dense_resource_ids(self.transfers, PlanError::DuplicateTransferId, |x| x.id)?;
        Self::validate_dense_resource_ids(
            self.transfer_batches,
            PlanError::DuplicateTransferBatchId,
            |x| x.id,
        )?;
        Self::validate_dense_resource_ids(
            self.synchronization,
            PlanError::DuplicateSyncId,
            |x| x.id,
        )?;
        self.validate_memory_group_canonicality()?;

        Ok(())
    }

}

// =============================================================================
// Canonical graph -> execution mapping validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_canonical_mapping(&self) -> Result<(), PlanError> {
        // Every execution node must identify one or more canonical semantic nodes.
        let mut e = 0usize;
        while e < self.nodes.len() {
            let exec = &self.nodes[e];
            if exec.semantic_nodes.is_empty() {
                return Err(PlanError::InvalidCanonicalGraph);
            }

            // Execution dependencies are a set semantically, so the frozen
            // encoding must use one canonical strictly-increasing order.
            let mut dependency_index = 1usize;
            while dependency_index < exec.dependencies.len() {
                if exec.dependencies[dependency_index - 1] >= exec.dependencies[dependency_index] {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                dependency_index += 1;
            }

            // Semantic ids inside one execution node must be unique.
            let mut i = 0usize;
            while i < exec.semantic_nodes.len() {
                let semantic_id = exec.semantic_nodes[i];
                let canonical = self
                    .canonical
                    .find_node(semantic_id)
                    .ok_or(PlanError::InvalidCanonicalGraph)?;
                let mut j = i + 1;
                while j < exec.semantic_nodes.len() {
                    if exec.semantic_nodes[j] == semantic_id {
                        return Err(PlanError::InvalidCanonicalGraph);
                    }
                    j += 1;
                }

                if exec.fused_operations.is_empty() {
                    if exec.semantic_nodes.len() != 1
                        || canonical.operation != exec.operation
                        || canonical.inputs != exec.inputs
                        || canonical.outputs != exec.outputs
                    {
                        return Err(PlanError::InvalidCanonicalGraph);
                    }
                } else {
                    if exec.semantic_nodes.len() != exec.fused_operations.len() {
                        return Err(PlanError::InvalidFusionDataflow);
                    }
                    let step = &exec.fused_operations[i];
                    if step.semantic_node != semantic_id
                        || step.operation != canonical.operation
                        || step.inputs != canonical.inputs
                        || step.outputs != canonical.outputs
                    {
                        return Err(PlanError::InvalidFusionDataflow);
                    }
                }
                i += 1;
            }

            // Exact execution dependencies are induced by canonical dependencies
            // that cross the current fusion boundary.
            i = 0;
            while i < exec.semantic_nodes.len() {
                let canonical = self
                    .canonical
                    .find_node(exec.semantic_nodes[i])
                    .ok_or(PlanError::InvalidCanonicalGraph)?;
                let mut d = 0usize;
                while d < canonical.dependencies.len() {
                    let dependency_semantic = canonical.dependencies[d];
                    let mut dependency_exec: Option<NodeId> = None;
                    let mut de = 0usize;
                    while de < self.nodes.len() {
                        if self.nodes[de].semantic_nodes.contains(&dependency_semantic) {
                            if dependency_exec.is_some() {
                                return Err(PlanError::InvalidCanonicalGraph);
                            }
                            dependency_exec = Some(self.nodes[de].id);
                        }
                        de += 1;
                    }
                    let dependency_exec = dependency_exec.ok_or(PlanError::InvalidCanonicalGraph)?;
                    if dependency_exec == exec.id {
                        // Internal fused dependency must point to an earlier
                        // semantic step in this same execution node.
                        let mut earlier = false;
                        let mut k = 0usize;
                        while k < i {
                            if exec.semantic_nodes[k] == dependency_semantic {
                                earlier = true;
                                break;
                            }
                            k += 1;
                        }
                        if !earlier {
                            return Err(PlanError::InvalidFusionDataflow);
                        }
                    } else if !exec.dependencies.contains(&dependency_exec) {
                        return Err(PlanError::MissingDataDependency);
                    }
                    d += 1;
                }
                i += 1;
            }

            // No execution dependency may exist without a canonical dependency
            // crossing this execution-node/fusion boundary.
            let mut d = 0usize;
            while d < exec.dependencies.len() {
                let declared = exec.dependencies[d];
                let mut justified = false;
                let mut si = 0usize;
                while si < exec.semantic_nodes.len() && !justified {
                    let canonical = self
                        .canonical
                        .find_node(exec.semantic_nodes[si])
                        .ok_or(PlanError::InvalidCanonicalGraph)?;
                    let mut cd = 0usize;
                    while cd < canonical.dependencies.len() && !justified {
                        let dependency_semantic = canonical.dependencies[cd];
                        let mut de = 0usize;
                        while de < self.nodes.len() {
                            if self.nodes[de].id == declared
                                && self.nodes[de].semantic_nodes.contains(&dependency_semantic)
                            {
                                justified = true;
                                break;
                            }
                            de += 1;
                        }
                        cd += 1;
                    }
                    si += 1;
                }
                if !justified {
                    return Err(PlanError::InvalidCanonicalGraph);
                }
                d += 1;
            }
            e += 1;
        }

        // Every canonical semantic node must be mapped exactly once globally.
        let mut c = 0usize;
        while c < self.canonical.graph.nodes.len() {
            let semantic_id = self.canonical.graph.nodes[c].id;
            let mut count = 0usize;
            let mut e = 0usize;
            while e < self.nodes.len() {
                let mut i = 0usize;
                while i < self.nodes[e].semantic_nodes.len() {
                    if self.nodes[e].semantic_nodes[i] == semantic_id {
                        count += 1;
                    }
                    i += 1;
                }
                e += 1;
            }
            if count != 1 {
                return Err(PlanError::InvalidCanonicalGraph);
            }
            c += 1;
        }
        Ok(())
    }
}

// =============================================================================
// Tensors validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_tensor_specs(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.tensors.len() {
            let tensor = &self.tensors[i];
            tensor.layout.validate_for_dtype(&tensor.shape, tensor.dtype)?;
            tensor
                .shape
                .byte_count(tensor.dtype)
                .ok_or(PlanError::TensorByteCountOverflow)?;
            if !tensor.tolerance.is_valid() {
                return Err(PlanError::InvalidAccuracyContract);
            }
            let abs_tol = f32::from_bits(tensor.tolerance.max_abs_error_f32_bits);
            let rel_tol = f32::from_bits(tensor.tolerance.max_rel_error_f32_bits);
            if tensor.accuracy == AccuracyContract::Exact && (abs_tol != 0.0 || rel_tol != 0.0) {
                return Err(PlanError::InvalidAccuracyContract);
            }
            if tensor.accuracy == AccuracyContract::QuantizedExactScale {
                return Err(PlanError::InvalidAccuracyContract);
            }

            match tensor.role {
                TensorRole::Data | TensorRole::Auxiliary => {
                    if tensor.dtype != DType::F32 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                TensorRole::Index => {
                    if !matches!(tensor.dtype, DType::I32 | DType::I64 | DType::U64) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    if tensor.accuracy != AccuracyContract::Exact || abs_tol != 0.0 || rel_tol != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
                TensorRole::IndexResult => {
                    if tensor.dtype != DType::U64 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    if tensor.accuracy != AccuracyContract::Exact || abs_tol != 0.0 || rel_tol != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
                TensorRole::Mask => {
                    if tensor.dtype != DType::U8 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    if tensor.accuracy != AccuracyContract::Exact || abs_tol != 0.0 || rel_tol != 0.0 {
                        return Err(PlanError::InvalidAccuracyContract);
                    }
                }
            }

            // The signed graph and the frozen plan MUST enforce the SAME
            // DENSE_ONLY release scope. Valid-looking sparse metadata is not
            // executable merely because the wire enum contains sparse tags.
            tensor.sparsity.validate_dense_only()?;

            let q = tensor.quantization;
            let rank = tensor.shape.rank();
            match q.kind {
                QuantizationKind::None => {
                    if q.granularity != ScaleGranularity::None
                        || q.axis0 != 0
                        || q.axis1 != 0
                        || q.scale_tensor.is_some()
                        || q.secondary_scale_tensor.is_some()
                        || q.zero_point_tensor.is_some()
                        || q.block0 != 0
                        || q.block1 != 0
                    {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                QuantizationKind::Symmetric
                | QuantizationKind::MxBlockScaled
                | QuantizationKind::NvFp4 => {
                    if q.scale_tensor.is_none() || q.granularity == ScaleGranularity::None {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                QuantizationKind::Asymmetric => {
                    if q.scale_tensor.is_none()
                        || q.zero_point_tensor.is_none()
                        || q.granularity == ScaleGranularity::None
                    {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                QuantizationKind::CustomStatic => {
                    if q.granularity == ScaleGranularity::None {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
            }
            // tensor_light value tensors are F32; engine-only index/mask
            // metadata may be integer/byte. Quantization remains disabled in
            // this concrete variant for every role.
            if q.kind != QuantizationKind::None {
                return Err(PlanError::InvalidQuantizationSpec);
            }
            match q.granularity {
                ScaleGranularity::None | ScaleGranularity::PerTensor => {}
                ScaleGranularity::PerAxis => {
                    if q.axis0 as usize >= rank {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                ScaleGranularity::Block1D => {
                    if q.axis0 as usize >= rank || q.block0 == 0 {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                ScaleGranularity::Block2D => {
                    if q.axis0 as usize >= rank
                        || q.axis1 as usize >= rank
                        || q.axis0 == q.axis1
                        || q.block0 == 0
                        || q.block1 == 0
                    {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
            }
            let refs = [q.scale_tensor, q.secondary_scale_tensor, q.zero_point_tensor];
            let mut r = 0usize;
            while r < refs.len() {
                if let Some(id) = refs[r] {
                    if id == tensor.id || self.find_tensor(id).is_none() {
                        return Err(PlanError::InvalidQuantizationSpec);
                    }
                }
                r += 1;
            }
            i += 1;
        }
        Ok(())
    }

}

// =============================================================================
// Operations validation
// =============================================================================

fn find_tensor_in_specs(tensors: &[TensorSpec], id: TensorId) -> Option<&TensorSpec> {
    tensors.get(id as usize).filter(|tensor| tensor.id == id)
}

#[inline(always)]
fn is_f32_value_tensor(tensor: &TensorSpec) -> bool {
    tensor.dtype == DType::F32
        && matches!(tensor.role, TensorRole::Data | TensorRole::Auxiliary)
}

#[inline(always)]
fn same_dtype_role(a: &TensorSpec, b: &TensorSpec) -> bool {
    a.dtype == b.dtype && a.role == b.role
}

pub(super) fn physical_minimum_bytes(
    layout: &TensorLayout,
    shape: &StaticShape,
    dtype: DType,
    base_offset_bytes: usize,
) -> Result<Option<usize>, PlanError> {
    let element_bytes = dtype.size_bytes();
    if !base_offset_bytes.is_multiple_of(element_bytes) {
        return Err(PlanError::InvalidPhysicalStorage);
    }
    match layout.kind {
        LayoutKind::RowMajorContiguous
        | LayoutKind::ColumnMajorContiguous
        | LayoutKind::ChannelsLast => {
            let bytes = shape
                .byte_count(dtype)
                .ok_or(PlanError::TensorByteCountOverflow)?;
            base_offset_bytes
                .checked_add(bytes)
                .map(Some)
                .ok_or(PlanError::InvalidPhysicalStorage)
        }
        LayoutKind::Strided => {
            let elements = shape
                .element_count()
                .ok_or(PlanError::TensorByteCountOverflow)?;
            if elements == 0 {
                return Ok(Some(base_offset_bytes));
            }

            let mut min_rel_elements = 0i128;
            let mut max_rel_elements = 0i128;
            let mut axis = 0usize;
            while axis < shape.rank() {
                let dim = shape.dim(axis).ok_or(PlanError::InvalidPhysicalStorage)?;
                if dim > 0 {
                    let extent = i128::try_from(dim - 1)
                        .map_err(|_| PlanError::InvalidPhysicalStorage)?;
                    let stride = layout.strides[axis] as i128;
                    let contribution = extent
                        .checked_mul(stride)
                        .ok_or(PlanError::InvalidPhysicalStorage)?;
                    if contribution < 0 {
                        min_rel_elements = min_rel_elements
                            .checked_add(contribution)
                            .ok_or(PlanError::InvalidPhysicalStorage)?;
                    } else {
                        max_rel_elements = max_rel_elements
                            .checked_add(contribution)
                            .ok_or(PlanError::InvalidPhysicalStorage)?;
                    }
                }
                axis += 1;
            }

            let element_bytes_i128 = element_bytes as i128;
            let base = base_offset_bytes as i128;
            let low = base
                .checked_add(
                    min_rel_elements
                        .checked_mul(element_bytes_i128)
                        .ok_or(PlanError::InvalidPhysicalStorage)?,
                )
                .ok_or(PlanError::InvalidPhysicalStorage)?;
            if low < 0 {
                return Err(PlanError::InvalidPhysicalStorage);
            }
            let high_exclusive = base
                .checked_add(
                    max_rel_elements
                        .checked_mul(element_bytes_i128)
                        .ok_or(PlanError::InvalidPhysicalStorage)?,
                )
                .and_then(|value| value.checked_add(element_bytes_i128))
                .ok_or(PlanError::InvalidPhysicalStorage)?;
            usize::try_from(high_exclusive)
                .map(Some)
                .map_err(|_| PlanError::InvalidPhysicalStorage)
        }
        LayoutKind::Blocked | LayoutKind::Packed | LayoutKind::TargetNative => {
            if base_offset_bytes != 0 {
                // Opaque target-native formats need a target-specific subview
                // descriptor before non-zero byte bases can be proven safe.
                return Err(PlanError::InvalidPhysicalStorage);
            }
            Ok(None)
        }
    }
}

pub(super) fn validate_writable_strided_nonoverlap(
    layout: &TensorLayout,
    shape: &StaticShape,
) -> Result<(), PlanError> {
    if layout.kind != LayoutKind::Strided {
        return Ok(());
    }

    // The physical writable-slot guard and the public host Tensor/ViewMut
    // constructors must use ONE sound injectivity certificate. The former
    // separated-stride-only shortcut rejected some valid writable layouts:
    // shape [3,2], strides [3,4] addresses 0,4,3,7,6,10 are distinct.
    //
    // This is checked only when validating an immutable plan's resource
    // bindings, never in a native element loop or as runtime scheduling.
    // `validate_for` and physical allocation/bounds checks remain separate
    // mandatory guards in validate_memory().
    let len = shape.element_count().ok_or(PlanError::TensorByteCountOverflow)?;
    let mut dims = [1usize; crate::tensor_light::tensor::MAX_RANK];
    let mut strides = [0isize; crate::tensor_light::tensor::MAX_RANK];
    for axis in 0..shape.rank() {
        dims[axis] = shape.dim(axis).ok_or(PlanError::InvalidPhysicalStorage)?;
        strides[axis] = layout.strides[axis];
    }

    // None is PROVEN injective. Definite and Unknown both reject a writable
    // alias: uncertainty is never a permission for overlapping &mut access.
    match crate::tensor_light::tensor::signed_mapping_overlap(&dims, &strides, len) {
        crate::tensor_light::tensor::InternalOverlap::None => Ok(()),
        crate::tensor_light::tensor::InternalOverlap::Definite
        | crate::tensor_light::tensor::InternalOverlap::Unknown => Err(PlanError::InvalidStorageAccess),
    }
}

fn convolution_output_extent(
    input: usize,
    kernel: u32,
    stride: u32,
    dilation: u32,
    padding_before: u32,
    padding_after: u32,
    transposed: bool,
) -> Result<usize, PlanError> {
    let kernel = usize::try_from(kernel).map_err(|_| PlanError::InvalidOperationSpec)?;
    let stride = usize::try_from(stride).map_err(|_| PlanError::InvalidOperationSpec)?;
    let dilation = usize::try_from(dilation).map_err(|_| PlanError::InvalidOperationSpec)?;
    let before = usize::try_from(padding_before).map_err(|_| PlanError::InvalidOperationSpec)?;
    let after = usize::try_from(padding_after).map_err(|_| PlanError::InvalidOperationSpec)?;
    let effective = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|v| v.checked_add(1))
        .ok_or(PlanError::InvalidOperationSpec)?;
    if transposed {
        if input == 0 {
            return Ok(0);
        }
        let expanded = (input - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(effective))
            .ok_or(PlanError::InvalidOperationSpec)?;
        let padding = before
            .checked_add(after)
            .ok_or(PlanError::InvalidOperationSpec)?;
        return expanded
            .checked_sub(padding)
            .ok_or(PlanError::InvalidOperationSpec);
    }
    let padded = input
        .checked_add(before)
        .and_then(|v| v.checked_add(after))
        .ok_or(PlanError::InvalidOperationSpec)?;
    if padded < effective {
        return Ok(0);
    }
    Ok((padded - effective) / stride + 1)
}

fn pooling_output_extent(
    input: usize,
    kernel: u32,
    stride: u32,
    dilation: u32,
    padding_before: u32,
    padding_after: u32,
    ceil_mode: bool,
) -> Result<usize, PlanError> {
    let kernel = usize::try_from(kernel).map_err(|_| PlanError::InvalidOperationSpec)?;
    let stride = usize::try_from(stride).map_err(|_| PlanError::InvalidOperationSpec)?;
    let dilation = usize::try_from(dilation).map_err(|_| PlanError::InvalidOperationSpec)?;
    let before = usize::try_from(padding_before).map_err(|_| PlanError::InvalidOperationSpec)?;
    let after = usize::try_from(padding_after).map_err(|_| PlanError::InvalidOperationSpec)?;
    let effective = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|v| v.checked_add(1))
        .ok_or(PlanError::InvalidOperationSpec)?;
    let padded = input
        .checked_add(before)
        .and_then(|v| v.checked_add(after))
        .ok_or(PlanError::InvalidOperationSpec)?;
    if padded < effective {
        return Ok(0);
    }
    let numerator = padded - effective;
    let mut output = if ceil_mode {
        numerator
            .checked_add(stride - 1)
            .ok_or(PlanError::InvalidOperationSpec)?
            / stride
            + 1
    } else {
        numerator / stride + 1
    };
    if ceil_mode && output > 0 {
        let last_start = (output - 1)
            .checked_mul(stride)
            .ok_or(PlanError::InvalidOperationSpec)?;
        let right_limit = input
            .checked_add(before)
            .ok_or(PlanError::InvalidOperationSpec)?;
        if last_start >= right_limit {
            output -= 1;
        }
    }
    Ok(output)
}

pub(super) fn validate_operation_io_specs(
    tensors: &[TensorSpec],
    operation: Operation,
    inputs: &[TensorId],
    outputs: &[TensorId],
) -> Result<(), PlanError> {
    operation.validate_basic()?;

    let input0 = if inputs.is_empty() {
        None
    } else {
        Some(find_tensor_in_specs(tensors, inputs[0]).ok_or(PlanError::MissingTensor)?)
    };
    let output0 = if outputs.is_empty() {
        None
    } else {
        Some(find_tensor_in_specs(tensors, outputs[0]).ok_or(PlanError::MissingTensor)?)
    };
    let require_input0 = || input0.ok_or(PlanError::InvalidOperationSpec);
    let require_output0 = || output0.ok_or(PlanError::InvalidOperationSpec);

    match operation {
        Operation::Copy => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if input.shape != output.shape || !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Cast => {
            // Exact conversion/rounding/saturation policy is not represented
            // by this light IR yet. Keep the opcode reserved, never ambiguous.
            return Err(PlanError::InvalidOperationSpec);
        }
        Operation::Reshape => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let lhs = require_input0()?
                .shape
                .element_count()
                .ok_or(PlanError::TensorByteCountOverflow)?;
            let rhs = require_output0()?
                .shape
                .element_count()
                .ok_or(PlanError::TensorByteCountOverflow)?;
            if lhs != rhs || !same_dtype_role(require_input0()?, require_output0()?) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Transpose { permutation } => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            if usize::from(permutation.len) != rank
                || output.shape.rank() != rank
                || !same_dtype_role(input, output)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            permutation.validate_for_rank(rank)?;
            let mut axis = 0usize;
            while axis < rank {
                let source_axis = usize::from(permutation.axes[axis]);
                if output.shape.dim(axis) != input.shape.dim(source_axis) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                axis += 1;
            }
        }
        Operation::Expand => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !same_dtype_role(input, output) || output.shape.rank() < input.shape.rank() {
                return Err(PlanError::InvalidOperationSpec);
            }
            let pad = output.shape.rank() - input.shape.rank();
            let mut axis = 0usize;
            while axis < output.shape.rank() {
                if axis >= pad {
                    let source = input
                        .shape
                        .dim(axis - pad)
                        .ok_or(PlanError::InvalidOperationSpec)?;
                    let target = output.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
                    if source != target && source != 1 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                axis += 1;
            }
        }
        Operation::AsStrided(_) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !same_dtype_role(input, output) || output.layout.kind != LayoutKind::Strided {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Narrow(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            let axis = usize::from(spec.axis);
            if rank == 0 || output.shape.rank() != rank || axis >= rank || !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let start = usize::try_from(spec.start).map_err(|_| PlanError::InvalidOperationSpec)?;
            let length = usize::try_from(spec.length).map_err(|_| PlanError::InvalidOperationSpec)?;
            let source_dim = input.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
            if start > source_dim || length > source_dim.saturating_sub(start) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut d = 0usize;
            while d < rank {
                let expected = if d == axis { length } else { input.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)? };
                if output.shape.dim(d) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::Flip { axes } => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            axes.validate_for_rank(rank)?;
            if input.shape != output.shape || !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            // An axis with extent zero or one has no pairs of elements to
            // reverse. Flip is a valid identity on that axis, including when
            // other axes remain nonempty. Keep the exact axis list in the IR
            // and let the OFFLINE plan choose its certified view/no-op mapping.
            // `axes.validate_for_rank` above still rejects duplicate and
            // out-of-range axes; it does not infer a new execution schedule.
        }
        Operation::Concat { axis } => {
            // One input is a valid identity concatenation. The exact graph
            // still requires its published alias/copy effects at replay.
            if inputs.is_empty() || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let first = require_input0()?;
            let output = require_output0()?;
            let rank = first.shape.rank();
            let axis = usize::from(axis);
            if rank == 0 || axis >= rank || output.shape.rank() != rank || !same_dtype_role(first, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut concat_dim = 0usize;
            let mut i = 0usize;
            while i < inputs.len() {
                let tensor = find_tensor_in_specs(tensors, inputs[i]).ok_or(PlanError::MissingTensor)?;
                if tensor.shape.rank() != rank || !same_dtype_role(first, tensor) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                let mut d = 0usize;
                while d < rank {
                    let dim = tensor.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)?;
                    if d == axis {
                        concat_dim = concat_dim.checked_add(dim).ok_or(PlanError::InvalidOperationSpec)?;
                    } else if dim != first.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)? {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    d += 1;
                }
                i += 1;
            }
            let mut d = 0usize;
            while d < rank {
                let expected = if d == axis { concat_dim } else { first.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)? };
                if output.shape.dim(d) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::Stack { axis } => {
            // One input still inserts a new singleton axis; it is not a
            // zero-input factory and must obey the rank<=8 ceiling.
            if inputs.is_empty() || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let first = require_input0()?;
            let output = require_output0()?;
            let rank = first.shape.rank();
            let axis = usize::from(axis);
            if rank >= crate::tensor_light::tensor::MAX_RANK || axis > rank || output.shape.rank() != rank + 1 || !same_dtype_role(first, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut i = 0usize;
            while i < inputs.len() {
                let tensor = find_tensor_in_specs(tensors, inputs[i]).ok_or(PlanError::MissingTensor)?;
                if tensor.shape != first.shape || !same_dtype_role(first, tensor) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                i += 1;
            }
            let mut od = 0usize;
            while od < rank + 1 {
                let expected = if od == axis {
                    inputs.len()
                } else {
                    let source_axis = if od < axis { od } else { od - 1 };
                    first.shape.dim(source_axis).ok_or(PlanError::InvalidOperationSpec)?
                };
                if output.shape.dim(od) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                od += 1;
            }
        }
        Operation::Pad(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            if output.shape.rank() != rank
                || !same_dtype_role(input, output)
                || input.dtype != DType::F32
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            // Rank zero has no axes. Exactly zero padding metadata is a
            // valid scalar identity; the loop still rejects nonzero fields
            // in EVERY unused slot, even for rank-zero inputs.
            let mut d = 0usize;
            while d < crate::tensor_light::tensor::MAX_RANK {
                if d >= rank {
                    if spec.before[d] != 0 || spec.after[d] != 0 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    d += 1;
                    continue;
                }
                let input_dim = input.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)?;
                let before = usize::try_from(spec.before[d]).map_err(|_| PlanError::InvalidOperationSpec)?;
                let after = usize::try_from(spec.after[d]).map_err(|_| PlanError::InvalidOperationSpec)?;
                let expected = input_dim
                    .checked_add(before)
                    .and_then(|v| v.checked_add(after))
                    .ok_or(PlanError::InvalidOperationSpec)?;
                if output.shape.dim(d) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                if before != 0 || after != 0 {
                    match spec.mode {
                        PadMode::Constant => {}
                        PadMode::Replicate | PadMode::Circular => {
                            if input_dim == 0 {
                                return Err(PlanError::InvalidOperationSpec);
                            }
                        }
                        PadMode::Reflect => {
                            if input_dim <= 1 || before >= input_dim || after >= input_dim {
                                return Err(PlanError::InvalidOperationSpec);
                            }
                        }
                    }
                }
                d += 1;
            }
            // All-zero padding is an exact semantic identity. The graph still
            // names Pad and any physical alias/copy/output publication remains
            // an OFFLINE-sealed effect; no eager runtime copy is authorized.
        }
        Operation::Resample2d(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            if rank < 2 || output.shape.rank() != rank || !same_dtype_role(input, output) || input.dtype != DType::F32 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let scale = usize::try_from(spec.scale).map_err(|_| PlanError::InvalidOperationSpec)?;
            if scale == 0 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let h_axis = rank - 2;
            let w_axis = rank - 1;
            let h = input.shape.dim(h_axis).ok_or(PlanError::InvalidOperationSpec)?;
            let w = input.shape.dim(w_axis).ok_or(PlanError::InvalidOperationSpec)?;
            let (out_h, out_w) = match spec.mode {
                ResampleMode::Nearest | ResampleMode::Bilinear => (
                    h.checked_mul(scale).ok_or(PlanError::InvalidOperationSpec)?,
                    w.checked_mul(scale).ok_or(PlanError::InvalidOperationSpec)?,
                ),
                ResampleMode::MeanDownsample => {
                    if h < scale || w < scale {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    (h / scale, w / scale)
                }
            };
            let mut d = 0usize;
            while d < rank {
                let expected = if d == h_axis { out_h } else if d == w_axis { out_w } else { input.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)? };
                if output.shape.dim(d) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::SlidingWindow1d(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if input.shape.rank() != 1 || output.shape.rank() != 2 || !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let n = input.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
            let window = usize::try_from(spec.window).map_err(|_| PlanError::InvalidOperationSpec)?;
            let step = usize::try_from(spec.step).map_err(|_| PlanError::InvalidOperationSpec)?;
            if window == 0 || step == 0 || window > n {
                return Err(PlanError::InvalidOperationSpec);
            }
            let rows = 1 + (n - window) / step;
            if output.shape.dim(0) != Some(rows) || output.shape.dim(1) != Some(window) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::ZeroInsert1d { factor } => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if input.dtype != DType::F32
                || input.shape.rank() != 1
                || output.shape.rank() != 1
                || !same_dtype_role(input, output)
                || factor == 0
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let n = input.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
            let factor = usize::try_from(factor).map_err(|_| PlanError::InvalidOperationSpec)?;
            // For n=0 there are no points between which zeros can be
            // inserted. n=1 or factor=1 preserves the input. Identity and
            // empty effects remain nodes of the EXACT frozen graph.
            let expected = if n == 0 {
                0
            } else {
                (n - 1)
                    .checked_mul(factor)
                    .and_then(|v| v.checked_add(1))
                    .ok_or(PlanError::InvalidOperationSpec)?
            };
            if output.shape.dim(0) != Some(expected) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Tile(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            let rank = input.shape.rank();
            if usize::from(spec.rank) != rank
                || output.shape.rank() != rank
                || !same_dtype_role(input, output)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut d = 0usize;
            while d < rank {
                let rep = usize::try_from(spec.repeats[d]).map_err(|_| PlanError::InvalidOperationSpec)?;
                let expected = input.shape.dim(d)
                    .ok_or(PlanError::InvalidOperationSpec)?
                    .checked_mul(rep)
                    .ok_or(PlanError::InvalidOperationSpec)?;
                if output.shape.dim(d) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::RepeatInterleave(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let repeats = usize::try_from(spec.repeats).map_err(|_| PlanError::InvalidOperationSpec)?;
            let rank = input.shape.rank();
            if rank == 0 {
                if spec.axis != 0 || output.shape.rank() != 1 || output.shape.dim(0) != Some(repeats) {
                    return Err(PlanError::InvalidOperationSpec);
                }
            } else {
                let axis = usize::from(spec.axis);
                if axis >= rank || output.shape.rank() != rank {
                    return Err(PlanError::InvalidOperationSpec);
                }
                let mut d = 0usize;
                while d < rank {
                    let mut expected = input.shape.dim(d).ok_or(PlanError::InvalidOperationSpec)?;
                    if d == axis {
                        expected = expected.checked_mul(repeats).ok_or(PlanError::InvalidOperationSpec)?;
                    }
                    if output.shape.dim(d) != Some(expected) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    d += 1;
                }
            }
        }
        Operation::Elementwise { kind, arity, broadcast_signature } => {
            if usize::from(arity) != inputs.len() || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let output = require_output0()?;
            let predicate = matches!(
                kind,
                ElementwiseKind::CompareEqual
                    | ElementwiseKind::CompareNotEqual
                    | ElementwiseKind::CompareLess
                    | ElementwiseKind::CompareLessEqual
                    | ElementwiseKind::CompareGreater
                    | ElementwiseKind::CompareGreaterEqual
                    | ElementwiseKind::IsFinite
                    | ElementwiseKind::IsNaN
                    | ElementwiseKind::IsInf
            );
            let logical = matches!(
                kind,
                ElementwiseKind::LogicalAnd
                    | ElementwiseKind::LogicalOr
                    | ElementwiseKind::LogicalNot
            );
            if ((predicate || logical)
                && (output.role != TensorRole::Mask || output.dtype != DType::U8))
                || (!predicate && !logical && !is_f32_value_tensor(output))
            {
                return Err(PlanError::InvalidOperationSpec);
            }

            let mut shapes = [StaticShape::Rank0; 2];
            let mut i = 0usize;
            while i < inputs.len() {
                let input = find_tensor_in_specs(tensors, inputs[i]).ok_or(PlanError::MissingTensor)?;
                let valid_input = if logical {
                    input.role == TensorRole::Mask && input.dtype == DType::U8
                } else {
                    is_f32_value_tensor(input)
                };
                if !valid_input {
                    return Err(PlanError::InvalidOperationSpec);
                }
                shapes[i] = input.shape;
                i += 1;
            }
            let expected_signature = canonical_elementwise_broadcast_signature(
                &shapes[..inputs.len()],
                output.shape,
            )?;
            if broadcast_signature != expected_signature {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::ScalarElementwise(_) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input)
                || !is_f32_value_tensor(output)
                || input.shape != output.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Select { broadcast_signature } => {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let condition = find_tensor_in_specs(tensors, inputs[0]).ok_or(PlanError::MissingTensor)?;
            let when_true = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let when_false = find_tensor_in_specs(tensors, inputs[2]).ok_or(PlanError::MissingTensor)?;
            let output = require_output0()?;
            if condition.role != TensorRole::Mask
                || condition.dtype != DType::U8
                || !is_f32_value_tensor(when_true)
                || !is_f32_value_tensor(when_false)
                || !is_f32_value_tensor(output)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let shapes = [condition.shape, when_true.shape, when_false.shape];
            let expected_signature = canonical_elementwise_broadcast_signature(&shapes, output.shape)?;
            if broadcast_signature != expected_signature {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Reduction { kind, axes, keep_dims } => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            match kind {
                ReductionKind::ArgMin | ReductionKind::ArgMax => {
                    if !is_f32_value_tensor(input)
                        || output.role != TensorRole::IndexResult
                        || output.dtype != DType::U64
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                ReductionKind::Any | ReductionKind::All => {
                    if input.role != TensorRole::Mask
                        || input.dtype != DType::U8
                        || output.role != TensorRole::Mask
                        || output.dtype != DType::U8
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                ReductionKind::AnyNonZero
                | ReductionKind::AllNonZero
                | ReductionKind::AnyPositive
                | ReductionKind::AllPositive => {
                    if !is_f32_value_tensor(input)
                        || output.role != TensorRole::Mask
                        || output.dtype != DType::U8
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                _ => {
                    if !is_f32_value_tensor(input) || !is_f32_value_tensor(output) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            }
            let input_rank = input.shape.rank();
            let axis_bits = axes.selection_mask(input_rank)?;
            let reduced = axis_bits.count_ones() as usize;
            // The canonical arg index is a single axis coordinate or a
            // flattened C-order index over ALL axes (rank0 -> index zero).
            // Explicit empty is not an arg reduction. Explicit multi-axis
            // arg reductions are outside this release's index ABI.
            if matches!(kind, ReductionKind::ArgMin | ReductionKind::ArgMax)
                && (axes.is_explicit_empty() ||
                    (matches!(axes, AxisSelector::Explicit(_)) && reduced != 1))
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
            let expected_rank = if keep_dims { input_rank } else { input_rank - reduced };
            if output.shape.rank() != expected_rank {
                return Err(PlanError::InvalidOperationSpec);
            }
            if keep_dims {
                for d in 0..input_rank {
                    let reduced_axis = (axis_bits & (1u8 << d)) != 0;
                    let expected = if reduced_axis { Some(1) } else { input.shape.dim(d) };
                    if output.shape.dim(d) != expected {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            } else {
                let mut out_axis = 0usize;
                for in_axis in 0..input_rank {
                    if (axis_bits & (1u8 << in_axis)) == 0 {
                        if output.shape.dim(out_axis) != input.shape.dim(in_axis) {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                        out_axis += 1;
                    }
                }
            }
            // Distinguish an output with zero elements from an evaluated empty
            // reduction domain. Only the latter needs the operator's identity
            // or empty-slice error semantics.
            let output_empty = output.shape.element_count() == Some(0);
            let mut empty_reduced_domain = false;
            for axis in 0..input_rank {
                if (axis_bits & (1u8 << axis)) != 0
                    && input.shape.dim(axis) == Some(0)
                {
                    empty_reduced_domain = true;
                }
            }
            // Every evaluated empty slice needs a PUBLISHED typed identity,
            // not merely a syntactically well-formed output shape. Besides
            // numeric sum/product, count-nonzero has +0.0f32 and the mask
            // OR/AND families have canonical false/true identities.
            // This is semantic validation of one static graph: no eager fill,
            // runtime kernel choice, fallback or implicit data movement.
            if empty_reduced_domain && !output_empty
                && kind.empty_reduction_identity().is_none()
            {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::MatMul { transpose_lhs, transpose_rhs } => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let lhs = find_tensor_in_specs(tensors, inputs[0]).ok_or(PlanError::MissingTensor)?;
            let rhs = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let out = require_output0()?;
            if !is_f32_value_tensor(lhs)
                || !is_f32_value_tensor(rhs)
                || !is_f32_value_tensor(out)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let lr = lhs.shape.rank();
            let rr = rhs.shape.rank();
            if lr == 0 || rr == 0
                || (lr == 1 && transpose_lhs)
                || (rr == 1 && transpose_rhs)
            {
                return Err(PlanError::InvalidOperationSpec);
            }

            let lhs_vector = lr == 1;
            let rhs_vector = rr == 1;

            // Conceptually promote lhs 1-D [K] -> [1,K] and rhs 1-D [K] -> [K,1],
            // apply ordinary batched matmul, then remove the promoted dimensions.
            let (lhs_rows, lhs_cols) = if lhs_vector {
                (1usize, lhs.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?)
            } else {
                (
                    lhs.shape.dim(lr - 2).ok_or(PlanError::InvalidOperationSpec)?,
                    lhs.shape.dim(lr - 1).ok_or(PlanError::InvalidOperationSpec)?,
                )
            };
            let (rhs_rows, rhs_cols) = if rhs_vector {
                (rhs.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?, 1usize)
            } else {
                (
                    rhs.shape.dim(rr - 2).ok_or(PlanError::InvalidOperationSpec)?,
                    rhs.shape.dim(rr - 1).ok_or(PlanError::InvalidOperationSpec)?,
                )
            };
            let (m, lhs_k) = if transpose_lhs {
                (lhs_cols, lhs_rows)
            } else {
                (lhs_rows, lhs_cols)
            };
            let (rhs_k, n) = if transpose_rhs {
                (rhs_cols, rhs_rows)
            } else {
                (rhs_rows, rhs_cols)
            };
            if lhs_k != rhs_k {
                return Err(PlanError::InvalidOperationSpec);
            }

            let lhs_batch_rank = if lhs_vector { 0 } else { lr - 2 };
            let rhs_batch_rank = if rhs_vector { 0 } else { rr - 2 };
            let batch_rank = lhs_batch_rank.max(rhs_batch_rank);
            let matrix_output_rank = (if lhs_vector { 0 } else { 1 })
                + (if rhs_vector { 0 } else { 1 });
            if out.shape.rank() != batch_rank + matrix_output_rank {
                return Err(PlanError::InvalidOperationSpec);
            }

            let mut b = 0usize;
            while b < batch_rank {
                let lhs_dim = if b + lhs_batch_rank >= batch_rank {
                    lhs.shape
                        .dim(b + lhs_batch_rank - batch_rank)
                        .unwrap_or(1)
                } else {
                    1
                };
                let rhs_dim = if b + rhs_batch_rank >= batch_rank {
                    rhs.shape
                        .dim(b + rhs_batch_rank - batch_rank)
                        .unwrap_or(1)
                } else {
                    1
                };
                let expected = canonical_broadcast_extent(lhs_dim, rhs_dim)
                    .ok_or(PlanError::InvalidOperationSpec)?;
                if out.shape.dim(b) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                b += 1;
            }

            let mut out_axis = batch_rank;
            if !lhs_vector {
                if out.shape.dim(out_axis) != Some(m) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                out_axis += 1;
            }
            if !rhs_vector && out.shape.dim(out_axis) != Some(n) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::FourierTransform(spec) => {
            let axis = usize::from(spec.axis);
            match spec.kind {
                FourierTransformKind::ComplexForward | FourierTransformKind::ComplexInverse => {
                    if inputs.len() != 2 || outputs.len() != 2 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let in_real = require_input0()?;
                    let in_imag = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
                    let out_real = require_output0()?;
                    let out_imag = find_tensor_in_specs(tensors, outputs[1]).ok_or(PlanError::MissingTensor)?;
                    if !is_f32_value_tensor(in_real)
                        || !is_f32_value_tensor(in_imag)
                        || !is_f32_value_tensor(out_real)
                        || !is_f32_value_tensor(out_imag)
                        || in_real.shape != in_imag.shape
                        || in_real.shape != out_real.shape
                        || in_real.shape != out_imag.shape
                        || in_real.role != in_imag.role
                        || in_real.role != out_real.role
                        || in_real.role != out_imag.role
                        || axis >= in_real.shape.rank()
                        || in_real.shape.dim(axis) == Some(0)
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                FourierTransformKind::RealForward => {
                    if inputs.len() != 1 || outputs.len() != 2 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let input = require_input0()?;
                    let out_real = require_output0()?;
                    let out_imag = find_tensor_in_specs(tensors, outputs[1]).ok_or(PlanError::MissingTensor)?;
                    if !is_f32_value_tensor(input)
                        || !is_f32_value_tensor(out_real)
                        || !is_f32_value_tensor(out_imag)
                        || out_real.shape != out_imag.shape
                        || input.role != out_real.role
                        || input.role != out_imag.role
                        || input.shape.rank() != out_real.shape.rank()
                        || axis >= input.shape.rank()
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let n = input.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
                    if n == 0 || out_real.shape.dim(axis) != Some(n / 2 + 1) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let mut d = 0usize;
                    while d < input.shape.rank() {
                        if d != axis && input.shape.dim(d) != out_real.shape.dim(d) {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                        d += 1;
                    }
                }
                FourierTransformKind::RealInverse => {
                    if inputs.len() != 2 || outputs.len() != 1 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let in_real = require_input0()?;
                    let in_imag = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
                    let output = require_output0()?;
                    if !is_f32_value_tensor(in_real)
                        || !is_f32_value_tensor(in_imag)
                        || !is_f32_value_tensor(output)
                        || in_real.shape != in_imag.shape
                        || in_real.role != in_imag.role
                        || in_real.role != output.role
                        || in_real.shape.rank() != output.shape.rank()
                        || axis >= output.shape.rank()
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let n = output.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
                    if n == 0 || in_real.shape.dim(axis) != Some(n / 2 + 1) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let mut d = 0usize;
                    while d < output.shape.rank() {
                        if d != axis && in_real.shape.dim(d) != output.shape.dim(d) {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                        d += 1;
                    }
                }
            }
        }
        Operation::DenseLinearAlgebra(spec) => {
            match spec.kind {
                DenseLinalgKind::Solve => {
                    if inputs.len() != 2 || outputs.len() != 1 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let a = require_input0()?;
                    let b = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
                    let output = require_output0()?;
                    let rank = a.shape.rank();
                    if !is_f32_value_tensor(a)
                        || !is_f32_value_tensor(b)
                        || !is_f32_value_tensor(output)
                        || rank < 2
                        || b.shape.rank() + 1 != rank
                        || output.shape != b.shape
                        || output.role != b.role
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let n = a.shape.dim(rank - 1).ok_or(PlanError::InvalidOperationSpec)?;
                    if n == 0 || a.shape.dim(rank - 2) != Some(n) || b.shape.dim(rank - 2) != Some(n) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let mut d = 0usize;
                    while d + 2 < rank {
                        if a.shape.dim(d) != b.shape.dim(d) {
                            return Err(PlanError::InvalidOperationSpec);
                        }
                        d += 1;
                    }
                }
                DenseLinalgKind::Inverse | DenseLinalgKind::CholeskyLower => {
                    if inputs.len() != 1 || outputs.len() != 1 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let input = require_input0()?;
                    let output = require_output0()?;
                    let rank = input.shape.rank();
                    if !is_f32_value_tensor(input)
                        || !is_f32_value_tensor(output)
                        || rank < 2
                        || input.shape != output.shape
                        || input.role != output.role
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let n = input.shape.dim(rank - 1).ok_or(PlanError::InvalidOperationSpec)?;
                    if n == 0 || input.shape.dim(rank - 2) != Some(n) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            }
        }
        Operation::TensorContract { lhs_axes, rhs_axes } => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let lhs = find_tensor_in_specs(tensors, inputs[0]).ok_or(PlanError::MissingTensor)?;
            let rhs = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let out = require_output0()?;
            if !is_f32_value_tensor(lhs)
                || !is_f32_value_tensor(rhs)
                || !is_f32_value_tensor(out)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            lhs_axes.validate_for_rank(lhs.shape.rank())?;
            rhs_axes.validate_for_rank(rhs.shape.rank())?;
            let contracted = usize::from(lhs_axes.len);
            let mut c = 0usize;
            while c < contracted {
                let la = usize::from(lhs_axes.axes[c]);
                let ra = usize::from(rhs_axes.axes[c]);
                if lhs.shape.dim(la) != rhs.shape.dim(ra) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                c += 1;
            }
            let expected_rank = lhs.shape.rank() + rhs.shape.rank() - 2 * contracted;
            if out.shape.rank() != expected_rank {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut out_axis = 0usize;
            let mut axis = 0usize;
            while axis < lhs.shape.rank() {
                let mut contracted_axis = false;
                let mut k = 0usize;
                while k < contracted {
                    if usize::from(lhs_axes.axes[k]) == axis { contracted_axis = true; break; }
                    k += 1;
                }
                if !contracted_axis {
                    if out.shape.dim(out_axis) != lhs.shape.dim(axis) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    out_axis += 1;
                }
                axis += 1;
            }
            axis = 0;
            while axis < rhs.shape.rank() {
                let mut contracted_axis = false;
                let mut k = 0usize;
                while k < contracted {
                    if usize::from(rhs_axes.axes[k]) == axis { contracted_axis = true; break; }
                    k += 1;
                }
                if !contracted_axis {
                    if out.shape.dim(out_axis) != rhs.shape.dim(axis) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    out_axis += 1;
                }
                axis += 1;
            }
        }
        Operation::Convolution(spec) => {
            if !(inputs.len() == 2 || inputs.len() == 3) || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let weight = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input)
                || !is_f32_value_tensor(weight)
                || !is_f32_value_tensor(output)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let spatial_rank = usize::from(spec.spatial_rank);
            let minimum_rank = spatial_rank + 2;
            let rank = input.shape.rank();
            if rank < minimum_rank
                || weight.shape.rank() != minimum_rank
                || output.shape.rank() != rank
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let channel_axis = rank - spatial_rank - 1;
            let input_channels = input.shape.dim(channel_axis).ok_or(PlanError::InvalidOperationSpec)?;
            let groups = usize::try_from(spec.groups).map_err(|_| PlanError::InvalidOperationSpec)?;
            if groups == 0 || input_channels % groups != 0 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let (weight_input_channels, output_channels) = if spec.transposed {
                let weight_in = weight.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
                let weight_out_per_group = weight.shape.dim(1).ok_or(PlanError::InvalidOperationSpec)?;
                let out = weight_out_per_group
                    .checked_mul(groups)
                    .ok_or(PlanError::InvalidOperationSpec)?;
                (weight_in, out)
            } else {
                let out = weight.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
                let weight_in_per_group = weight.shape.dim(1).ok_or(PlanError::InvalidOperationSpec)?;
                let expected_in = weight_in_per_group
                    .checked_mul(groups)
                    .ok_or(PlanError::InvalidOperationSpec)?;
                (expected_in, out)
            };
            if weight_input_channels != input_channels
                || output_channels % groups != 0
                || output.shape.dim(channel_axis) != Some(output_channels)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut prefix = 0usize;
            while prefix < channel_axis {
                if output.shape.dim(prefix) != input.shape.dim(prefix) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                prefix += 1;
            }
            let mut spatial = 0usize;
            while spatial < spatial_rank {
                let input_axis = rank - spatial_rank + spatial;
                let weight_axis = 2 + spatial;
                let input_extent = input.shape.dim(input_axis).ok_or(PlanError::InvalidOperationSpec)?;
                if weight.shape.dim(weight_axis) != Some(spec.kernel[spatial] as usize) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                let expected = convolution_output_extent(
                    input_extent,
                    spec.kernel[spatial],
                    spec.stride[spatial],
                    spec.dilation[spatial],
                    spec.padding_before[spatial],
                    spec.padding_after[spatial],
                    spec.transposed,
                )?;
                if output.shape.dim(input_axis) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                spatial += 1;
            }
            if inputs.len() == 3 {
                let bias = find_tensor_in_specs(tensors, inputs[2]).ok_or(PlanError::MissingTensor)?;
                if !is_f32_value_tensor(bias)
                    || bias.shape.rank() != 1
                    || bias.shape.dim(0) != Some(output_channels)
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
            }
        }
        Operation::Attention(spec) => {
            let needs_mask = matches!(spec.mask, AttentionMaskKind::Explicit | AttentionMaskKind::BlockSparse);
            let expected_inputs = if needs_mask { 4 } else { 3 };
            if inputs.len() != expected_inputs || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let query = require_input0()?;
            let key = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let value = find_tensor_in_specs(tensors, inputs[2]).ok_or(PlanError::MissingTensor)?;
            let output = require_output0()?;
            if !is_f32_value_tensor(query)
                || !is_f32_value_tensor(key)
                || !is_f32_value_tensor(value)
                || !is_f32_value_tensor(output)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let rank = query.shape.rank();
            if rank < 3
                || key.shape.rank() != rank
                || value.shape.rank() != rank
                || output.shape != query.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let head_axis = rank - 3;
            let sequence_axis = rank - 2;
            let dim_axis = rank - 1;
            let head_dim = usize::try_from(spec.head_dim).map_err(|_| PlanError::InvalidOperationSpec)?;
            let query_heads = usize::from(spec.query_heads);
            let kv_heads = usize::from(spec.kv_heads);
            if query.shape.dim(head_axis) != Some(query_heads)
                || key.shape.dim(head_axis) != Some(kv_heads)
                || value.shape.dim(head_axis) != Some(kv_heads)
                || query.shape.dim(dim_axis) != Some(head_dim)
                || key.shape.dim(dim_axis) != Some(head_dim)
                || value.shape.dim(dim_axis) != Some(head_dim)
                || key.shape.dim(sequence_axis) != value.shape.dim(sequence_axis)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut prefix = 0usize;
            while prefix < head_axis {
                if query.shape.dim(prefix) != key.shape.dim(prefix)
                    || query.shape.dim(prefix) != value.shape.dim(prefix)
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
                prefix += 1;
            }
            if needs_mask {
                let mask = find_tensor_in_specs(tensors, inputs[3]).ok_or(PlanError::MissingTensor)?;
                if mask.role != TensorRole::Mask || mask.dtype != DType::U8 || mask.shape.rank() > rank {
                    return Err(PlanError::InvalidOperationSpec);
                }
                if spec.mask == AttentionMaskKind::BlockSparse
                    && mask.sparsity.pattern != SparsityPattern::BlockSparse
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
                let mask_rank = mask.shape.rank();
                let mut trailing = 0usize;
                while trailing < mask_rank {
                    let mask_axis = mask_rank - 1 - trailing;
                    let score_axis = rank - 1 - trailing;
                    let target = if score_axis < head_axis {
                        query.shape.dim(score_axis).ok_or(PlanError::InvalidOperationSpec)?
                    } else if score_axis == head_axis {
                        query_heads
                    } else if score_axis == sequence_axis {
                        query.shape.dim(sequence_axis).ok_or(PlanError::InvalidOperationSpec)?
                    } else {
                        key.shape.dim(sequence_axis).ok_or(PlanError::InvalidOperationSpec)?
                    };
                    let dim = mask.shape.dim(mask_axis).ok_or(PlanError::InvalidOperationSpec)?;
                    if dim != 1 && dim != target {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    trailing += 1;
                }
            }
        }
        Operation::Collective(_) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if input.shape != output.shape || !same_dtype_role(input, output) {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Normalization(spec) => {
            // Affine parameters are represented as separate elementwise nodes
            // and can still be fused into the same target artifact offline.
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input) || !is_f32_value_tensor(output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let rank = input.shape.rank();
            spec.axes.validate_for_rank(rank)?;
            if output.shape != input.shape {
                return Err(PlanError::InvalidOperationSpec);
            }
            // Statistics over an empty normalized axis are undefined and differ
            // between backend libraries. Keep the frozen semantic contract exact.
            let mut normalized_axis = 0usize;
            while normalized_axis < rank {
                if (spec.axes.0 & (1u8 << normalized_axis)) != 0
                    && input.shape.dim(normalized_axis) == Some(0)
                {
                    return Err(PlanError::InvalidOperationSpec);
                }
                normalized_axis += 1;
            }

            match spec.kind {
                NormalizationKind::LayerNorm
                | NormalizationKind::RmsNorm
                | NormalizationKind::L2Norm => {}
                NormalizationKind::BatchNorm => {
                    let channel_axis = usize::from(spec.channel_axis);
                    // Canonical tensor_light convention: axis 0 is the sample/batch axis.
                    if rank < 2 || channel_axis == 0 || channel_axis >= rank {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    // Population statistics are computed over every non-channel axis,
                    // including the sample axis.
                    let valid_mask = if rank == crate::tensor_light::tensor::MAX_RANK {
                        u8::MAX
                    } else {
                        (1u8 << rank) - 1
                    };
                    let expected_axes = valid_mask & !(1u8 << channel_axis);
                    if spec.axes.0 != expected_axes {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                NormalizationKind::GroupNorm => {
                    let channel_axis = usize::from(spec.channel_axis);
                    // Group statistics are per sample and span channel + spatial axes.
                    if rank < 2 || channel_axis == 0 || channel_axis >= rank {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let valid_mask = if rank == crate::tensor_light::tensor::MAX_RANK {
                        u8::MAX
                    } else {
                        (1u8 << rank) - 1
                    };
                    if spec.axes.0 != (valid_mask & !1u8) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let channels = input
                        .shape
                        .dim(channel_axis)
                        .ok_or(PlanError::InvalidOperationSpec)?;
                    let groups = usize::try_from(spec.groups)
                        .map_err(|_| PlanError::InvalidOperationSpec)?;
                    if channels == 0 || groups == 0 || channels % groups != 0 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                NormalizationKind::InstanceNorm => {
                    let channel_axis = usize::from(spec.channel_axis);
                    // Instance statistics are per sample and per channel, so only
                    // spatial axes participate in the population statistics.
                    if rank < 2 || channel_axis == 0 || channel_axis >= rank {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let valid_mask = if rank == crate::tensor_light::tensor::MAX_RANK {
                        u8::MAX
                    } else {
                        (1u8 << rank) - 1
                    };
                    let expected_axes = valid_mask & !1u8 & !(1u8 << channel_axis);
                    if expected_axes == 0 || spec.axes.0 != expected_axes {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    let channels = input
                        .shape
                        .dim(channel_axis)
                        .ok_or(PlanError::InvalidOperationSpec)?;
                    if channels == 0 {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            }
        }
        Operation::Gather { axis } => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let source = require_input0()?;
            let indices = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let output = require_output0()?;
            let axis = usize::from(axis);
            if !is_f32_value_tensor(source)
                || !is_f32_value_tensor(output)
                || !same_dtype_role(source, output)
                || axis >= source.shape.rank()
                || indices.role != TensorRole::Index
                || !matches!(indices.dtype, DType::I32 | DType::I64 | DType::U64)
                || indices.shape.rank() != 1
                || output.shape.rank() != source.shape.rank()
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let index_len = indices.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
            let mut d = 0usize;
            while d < source.shape.rank() {
                let expected = if d == axis { Some(index_len) } else { source.shape.dim(d) };
                if output.shape.dim(d) != expected {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::Scatter(spec) => {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let base = require_input0()?;
            let indices = find_tensor_in_specs(tensors, inputs[1]).ok_or(PlanError::MissingTensor)?;
            let values = find_tensor_in_specs(tensors, inputs[2]).ok_or(PlanError::MissingTensor)?;
            let output = require_output0()?;
            let axis = usize::from(spec.axis);
            if !is_f32_value_tensor(base)
                || !is_f32_value_tensor(values)
                || !is_f32_value_tensor(output)
                || !same_dtype_role(base, values)
                || !same_dtype_role(base, output)
                || axis >= base.shape.rank()
                || indices.role != TensorRole::Index
                || !matches!(indices.dtype, DType::I32 | DType::I64 | DType::U64)
                || indices.shape.rank() != 1
                || output.shape != base.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            if values.shape.rank() != 0 {
                if values.shape.rank() != base.shape.rank() {
                    return Err(PlanError::InvalidOperationSpec);
                }
                let index_len = indices.shape.dim(0).ok_or(PlanError::InvalidOperationSpec)?;
                let mut d = 0usize;
                while d < base.shape.rank() {
                    let expected = if d == axis { Some(index_len) } else { base.shape.dim(d) };
                    if values.shape.dim(d) != expected {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                    d += 1;
                }
            }
        }
        Operation::Pooling(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input) || !is_f32_value_tensor(output) {
                return Err(PlanError::InvalidOperationSpec);
            }
            let spatial_rank = usize::from(spec.spatial_rank);
            let rank = input.shape.rank();
            if rank < spatial_rank + 2 || output.shape.rank() != rank {
                return Err(PlanError::InvalidOperationSpec);
            }
            let first_spatial = rank - spatial_rank;
            let mut leading = 0usize;
            while leading < first_spatial {
                if output.shape.dim(leading) != input.shape.dim(leading) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                leading += 1;
            }
            let mut spatial = 0usize;
            while spatial < spatial_rank {
                let axis = first_spatial + spatial;
                let input_extent = input.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
                let expected = pooling_output_extent(
                    input_extent,
                    spec.kernel[spatial],
                    spec.stride[spatial],
                    spec.dilation[spatial],
                    spec.padding_before[spatial],
                    spec.padding_after[spatial],
                    spec.ceil_mode,
                )?;
                if output.shape.dim(axis) != Some(expected) {
                    return Err(PlanError::InvalidOperationSpec);
                }
                spatial += 1;
            }
        }
        Operation::Softmax { axis, log: _ } => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input)
                || !is_f32_value_tensor(output)
                || usize::from(axis) >= input.shape.rank()
                || input.shape != output.shape
                || input.shape.dim(usize::from(axis)) == Some(0)
            {
                return Err(PlanError::InvalidOperationSpec);
            }
        }
        Operation::Scan(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input)
                || usize::from(spec.axis) >= input.shape.rank()
                || input.shape != output.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            match spec.output {
                ScanOutput::Values => {
                    if !is_f32_value_tensor(output) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                ScanOutput::Indices => {
                    if output.role != TensorRole::IndexResult
                        || output.dtype != DType::U64
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            }
        }
        Operation::Sort(spec) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let output = require_output0()?;
            if !is_f32_value_tensor(input)
                || usize::from(spec.axis) >= input.shape.rank()
                || input.shape != output.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            match spec.output {
                SortOutput::Values => {
                    if !is_f32_value_tensor(output) {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
                SortOutput::Indices => {
                    if output.role != TensorRole::IndexResult
                        || output.dtype != DType::U64
                    {
                        return Err(PlanError::InvalidOperationSpec);
                    }
                }
            }
        }
        Operation::TopK(spec) => {
            if inputs.len() != 1 || outputs.len() != 2 {
                return Err(PlanError::InvalidOperationSpec);
            }
            let input = require_input0()?;
            let values = find_tensor_in_specs(tensors, outputs[0]).ok_or(PlanError::MissingTensor)?;
            let indices = find_tensor_in_specs(tensors, outputs[1]).ok_or(PlanError::MissingTensor)?;
            let axis = usize::from(spec.axis);
            if !is_f32_value_tensor(input)
                || !is_f32_value_tensor(values)
                || indices.role != TensorRole::IndexResult
                || indices.dtype != DType::U64
                || axis >= input.shape.rank()
                || values.shape.rank() != input.shape.rank()
                || indices.shape != values.shape
            {
                return Err(PlanError::InvalidOperationSpec);
            }
            let axis_extent = input.shape.dim(axis).ok_or(PlanError::InvalidOperationSpec)?;
            let k = usize::try_from(spec.k).map_err(|_| PlanError::InvalidOperationSpec)?;
            if axis_extent == 0 || k == 0 || k > axis_extent {
                return Err(PlanError::InvalidOperationSpec);
            }
            let mut d = 0usize;
            while d < input.shape.rank() {
                let expected = if d == axis { Some(k) } else { input.shape.dim(d) };
                if values.shape.dim(d) != expected {
                    return Err(PlanError::InvalidOperationSpec);
                }
                d += 1;
            }
        }
        Operation::Custom { .. } => {}
    }
    Ok(())
}

impl FrozenExecutionPlan {
    pub(super) fn validate_operation_io(
        &self,
        operation: Operation,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> Result<(), PlanError> {
        validate_operation_io_specs(self.tensors, operation, inputs, outputs)
    }

    pub(super) fn validate_fused_dataflow(&self, node: &FrozenNode) -> Result<(), PlanError> {
        if node.fused_operations.is_empty() {
            return self.validate_operation_io(node.operation, node.inputs, node.outputs);
        }
        if node.fused_operations[0].operation != node.operation {
            return Err(PlanError::InvalidFusionDataflow);
        }

        let mut step_index = 0usize;
        while step_index < node.fused_operations.len() {
            let step = &node.fused_operations[step_index];
            self.validate_operation_io(step.operation, step.inputs, step.outputs)?;

            // Every consumed value is either an external node input or was
            // produced by an earlier step in this fused region.
            let mut input_index = 0usize;
            while input_index < step.inputs.len() {
                let tensor = step.inputs[input_index];
                self.find_tensor(tensor).ok_or(PlanError::MissingTensor)?;
                let mut available = node.inputs.contains(&tensor);
                if !available {
                    let mut earlier = 0usize;
                    while earlier < step_index && !available {
                        available = node.fused_operations[earlier].outputs.contains(&tensor);
                        earlier += 1;
                    }
                }
                if !available {
                    return Err(PlanError::InvalidFusionDataflow);
                }
                input_index += 1;
            }

            // Logical SSA inside a fused region: each output is unique and may
            // not overwrite an external logical input. Physical aliasing belongs
            // only to the memory plan.
            let mut output_index = 0usize;
            while output_index < step.outputs.len() {
                let tensor = step.outputs[output_index];
                self.find_tensor(tensor).ok_or(PlanError::MissingTensor)?;
                if node.inputs.contains(&tensor) {
                    return Err(PlanError::InvalidFusionDataflow);
                }
                let mut prior = 0usize;
                while prior <= step_index {
                    let limit = if prior == step_index { output_index } else { node.fused_operations[prior].outputs.len() };
                    let mut prior_output = 0usize;
                    while prior_output < limit {
                        if node.fused_operations[prior].outputs[prior_output] == tensor {
                            return Err(PlanError::InvalidFusionDataflow);
                        }
                        prior_output += 1;
                    }
                    prior += 1;
                }
                output_index += 1;
            }
            step_index += 1;
        }

        // The outer input list is exact: every declared input participates in
        // at least one fused operation.
        let mut input_index = 0usize;
        while input_index < node.inputs.len() {
            let tensor = node.inputs[input_index];
            let mut used = false;
            let mut step = 0usize;
            while step < node.fused_operations.len() && !used {
                used = node.fused_operations[step].inputs.contains(&tensor);
                step += 1;
            }
            if !used {
                return Err(PlanError::InvalidFusionDataflow);
            }
            input_index += 1;
        }

        // Every declared node output is produced by the fused region, and every
        // produced value that is not consumed later must escape as a node output.
        let mut output_index = 0usize;
        while output_index < node.outputs.len() {
            let tensor = node.outputs[output_index];
            let mut produced = false;
            let mut step = 0usize;
            while step < node.fused_operations.len() && !produced {
                produced = node.fused_operations[step].outputs.contains(&tensor);
                step += 1;
            }
            if !produced {
                return Err(PlanError::InvalidFusionDataflow);
            }
            output_index += 1;
        }

        step_index = 0;
        while step_index < node.fused_operations.len() {
            let step = &node.fused_operations[step_index];
            let mut produced_index = 0usize;
            while produced_index < step.outputs.len() {
                let tensor = step.outputs[produced_index];
                let mut consumed_later = false;
                let mut later = step_index + 1;
                while later < node.fused_operations.len() && !consumed_later {
                    consumed_later = node.fused_operations[later].inputs.contains(&tensor);
                    later += 1;
                }
                if !consumed_later && !node.outputs.contains(&tensor) {
                    return Err(PlanError::InvalidFusionDataflow);
                }
                produced_index += 1;
            }
            step_index += 1;
        }

        Ok(())
    }

    pub(super) fn validate_node_operation(&self, node: &FrozenNode) -> Result<(), PlanError> {
        self.validate_fused_dataflow(node)
    }

    pub(super) fn validate_nodes(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.nodes.len() {
            let node = &self.nodes[i];
            self.validate_node_operation(node)?;

            let node_start = self
                .node_schedule_start_position(node.id)
                .ok_or(PlanError::MissingScheduleItem)?;

            let mut d = 0usize;
            while d < node.dependencies.len() {
                let dep_id = node.dependencies[d];
                if !self.has_node(dep_id) {
                    return Err(PlanError::MissingDependencyNode);
                }
                let dep_completion = self
                    .node_schedule_completion_position(dep_id)
                    .ok_or(PlanError::MissingScheduleItem)?;
                if dep_completion >= node_start {
                    return Err(PlanError::InvalidDependencyOrder);
                }
                d += 1;
            }

            let mut v = 0usize;
            while v < node.inputs.len() {
                let tensor_id = node.inputs[v];
                self.find_tensor(tensor_id).ok_or(PlanError::MissingTensor)?;
                if let Some(producer) = self.tensor_producer(tensor_id) {
                    if producer == node.id {
                        return Err(PlanError::InvalidDependencyOrder);
                    }
                    let producer_completion = self
                        .node_schedule_completion_position(producer)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    if producer_completion >= node_start {
                        return Err(PlanError::InvalidDependencyOrder);
                    }
                    let mut declared = false;
                    let mut dependency_index = 0usize;
                    while dependency_index < node.dependencies.len() {
                        if node.dependencies[dependency_index] == producer {
                            declared = true;
                            break;
                        }
                        dependency_index += 1;
                    }
                    if !declared {
                        return Err(PlanError::MissingDataDependency);
                    }
                }
                v += 1;
            }
            v = 0;
            while v < node.outputs.len() {
                self.find_tensor(node.outputs[v]).ok_or(PlanError::MissingTensor)?;
                v += 1;
            }

            match node.placement {
                Placement::Cpu => {
                    if !self.canonical.execution_topology.cpu_enabled {
                        return Err(PlanError::InvalidExecutionTopology);
                    }
                    let cpu = node.cpu.ok_or(PlanError::MissingCpuMapping)?;
                    if node.gpu.is_some() || node.cpu_gpu.is_some() {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    self.validate_cpu_mapping(node, cpu)?;
                    self.validate_cpu_io(node, cpu)?;
                }
                Placement::Gpu => {
                    let gpu = node.gpu.ok_or(PlanError::MissingGpuMapping)?;
                    if !self.canonical.execution_topology.gpu_device_ordinals.contains(&gpu.device_ordinal) {
                        return Err(PlanError::InvalidExecutionTopology);
                    }
                    if node.cpu.is_some() || node.cpu_gpu.is_some() {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    self.validate_gpu_mapping(node, gpu)?;
                    self.validate_gpu_io(node, gpu)?;
                }
                Placement::CpuGpu => {
                    let cpu = node.cpu.ok_or(PlanError::MissingCpuMapping)?;
                    let gpu = node.gpu.ok_or(PlanError::MissingGpuMapping)?;
                    let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                    if !self.canonical.execution_topology.cpu_enabled
                        || !self.canonical.execution_topology.gpu_device_ordinals.contains(&gpu.device_ordinal)
                        || (mapping.mode == CpuGpuMode::Partitioned
                            && !self.canonical.execution_topology.allow_cpu_gpu_partitioned)
                        || (mapping.mode == CpuGpuMode::Pipeline
                            && !self.canonical.execution_topology.allow_cpu_gpu_pipeline)
                    {
                        return Err(PlanError::InvalidExecutionTopology);
                    }
                    self.validate_cpu_mapping(node, cpu)?;
                    self.validate_gpu_mapping(node, gpu)?;
                    self.validate_cpu_io(node, cpu)?;
                    self.validate_gpu_io(node, gpu)?;

                    if mapping.pipeline_depth == 0 {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    if mapping.slices.is_empty() {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    match mapping.mode {
                        CpuGpuMode::Partitioned => {
                            if mapping.pipeline_depth != 1 {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                        }
                        CpuGpuMode::Pipeline => {
                            if mapping.pipeline_depth < 2 || !mapping.overlap_cpu_gpu {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                        }
                    }
                    let mut has_cpu_slice = false;
                    let mut has_gpu_slice = false;
                    let mut s = 0usize;
                    while s < mapping.slices.len() {
                        let slice = &mapping.slices[s];
                        if slice.work.begin >= slice.work.end {
                            return Err(PlanError::InvalidWorkRange);
                        }
                        if slice.stage >= mapping.pipeline_depth
                            || (mapping.mode != CpuGpuMode::Pipeline && slice.stage != 0)
                        {
                            return Err(PlanError::InvalidCpuGpuMapping);
                        }
                        match slice.target {
                            ExecutionTarget::Cpu => has_cpu_slice = true,
                            ExecutionTarget::Gpu => has_gpu_slice = true,
                        }
                        if mapping.mode == CpuGpuMode::Partitioned {
                            let mut t = s + 1;
                            while t < mapping.slices.len() {
                                let other = &mapping.slices[t];
                                if slice.work.begin < other.work.end
                                    && other.work.begin < slice.work.end
                                {
                                    return Err(PlanError::InvalidCpuGpuMapping);
                                }
                                t += 1;
                            }
                        }
                        s += 1;
                    }
                    if !has_cpu_slice || !has_gpu_slice {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    if mapping.mode == CpuGpuMode::Pipeline {
                        let mut stage = 0u16;
                        while stage < mapping.pipeline_depth {
                            let mut present = false;
                            let mut slice_index = 0usize;
                            while slice_index < mapping.slices.len() {
                                if mapping.slices[slice_index].stage == stage {
                                    present = true;
                                    break;
                                }
                                slice_index += 1;
                            }
                            if !present {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                            stage += 1;
                        }
                    }
                    // A combined placement is a complete mapping of one canonical
                    // tensor node. All outputs therefore share the exact partition
                    // extent used by the frozen slices; no backend may infer coverage.
                    if node.outputs.is_empty() {
                        return Err(PlanError::InvalidCpuGpuPartitionAxis);
                    }
                    let first_output = self
                        .find_tensor(node.outputs[0])
                        .ok_or(PlanError::MissingTensor)?;
                    let rank = first_output.shape.rank();
                    if rank == 0 || mapping.partition_axis as usize >= rank {
                        return Err(PlanError::InvalidCpuGpuPartitionAxis);
                    }
                    let extent = first_output
                        .shape
                        .dim(mapping.partition_axis as usize)
                        .ok_or(PlanError::InvalidCpuGpuPartitionAxis)? as u64;
                    let mut output_index = 1usize;
                    while output_index < node.outputs.len() {
                        let output = self
                            .find_tensor(node.outputs[output_index])
                            .ok_or(PlanError::MissingTensor)?;
                        if output.shape.rank() <= mapping.partition_axis as usize
                            || output
                                .shape
                                .dim(mapping.partition_axis as usize)
                                .ok_or(PlanError::InvalidCpuGpuPartitionAxis)? as u64
                                != extent
                        {
                            return Err(PlanError::InvalidCpuGpuPartitionAxis);
                        }
                        output_index += 1;
                    }

                    match mapping.mode {
                        CpuGpuMode::Partitioned => {
                            let mut expected_begin = 0u64;
                            let mut slice_index = 0usize;
                            while slice_index < mapping.slices.len() {
                                let work = mapping.slices[slice_index].work;
                                if work.begin != expected_begin || work.end > extent {
                                    return Err(PlanError::InvalidCpuGpuMapping);
                                }
                                expected_begin = work.end;
                                slice_index += 1;
                            }
                            if expected_begin != extent {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                        }
                        CpuGpuMode::Pipeline => {
                            // Every pipeline stage is itself a complete, non-overlapping
                            // partition of the same canonical output extent. Matching work
                            // ranges across adjacent stages form the static pipeline edges.
                            let mut stage = 0u16;
                            while stage < mapping.pipeline_depth {
                                let mut expected_begin = 0u64;
                                let mut slice_index = 0usize;
                                while slice_index < mapping.slices.len() {
                                    let slice = mapping.slices[slice_index];
                                    if slice.stage == stage {
                                        if slice.work.begin != expected_begin || slice.work.end > extent {
                                            return Err(PlanError::InvalidCpuGpuMapping);
                                        }
                                        expected_begin = slice.work.end;
                                    }
                                    slice_index += 1;
                                }
                                if expected_begin != extent {
                                    return Err(PlanError::InvalidCpuGpuMapping);
                                }
                                stage += 1;
                            }
                        }
                    }
                }
            }

            i += 1;
        }

        Ok(())
    }

}

}

/// Shared canonical/frozen-plan operation shape/type validation. This runs
/// before offline candidate generation and again after execution mapping.
pub(super) fn validate_canonical_operation_io(
    tensors: &[TensorSpec],
    operation: Operation,
    inputs: &[TensorId],
    outputs: &[TensorId],
) -> Result<(), PlanError> {
    semantic::validate_operation_io_specs(tensors, operation, inputs, outputs)
}

mod placement {
    use super::*;

// =============================================================================
// Mapping Common validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_kernel_assumptions(
        &self,
        io: TargetIoPlan,
        assumptions: KernelAssumptionPlan,
    ) -> Result<(), PlanError> {
        if assumptions.aligned_access_bytes > 0
            && (!assumptions.aligned_access_bytes.is_power_of_two())
        {
            return Err(PlanError::InvalidKernelAssumption);
        }

        let total = io.input_storages.len() + io.output_storages.len();
        let get = |idx: usize| -> StorageId {
            if idx < io.input_storages.len() {
                io.input_storages[idx]
            } else {
                io.output_storages[idx - io.input_storages.len()]
            }
        };

        let mut i = 0usize;
        while i < total {
            let storage = self.find_storage(get(i)).ok_or(PlanError::MissingStorage)?;
            let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
            if assumptions.aligned_access_bytes > 0 {
                let alignment = assumptions.aligned_access_bytes as usize;
                let storage_address = slot
                    .offset_bytes
                    .checked_add(storage.base_offset_bytes)
                    .ok_or(PlanError::InvalidKernelAssumption)?;
                if slot.alignment_bytes < alignment
                    || storage_address % alignment != 0
                    || (storage.physical_layout.alignment_bytes as usize) < alignment
                {
                    return Err(PlanError::InvalidKernelAssumption);
                }
            }
            if assumptions.noalias_io {
                let mut j = i + 1;
                while j < total {
                    if self.storage_ranges_overlap(get(i), get(j))? {
                        return Err(PlanError::InvalidKernelAssumption);
                    }
                    j += 1;
                }
            }
            i += 1;
        }

        if (assumptions.unchecked_indexing || assumptions.assert_unchecked_invariants)
            && self.seal.verification_hash_hi == 0 && self.seal.verification_hash_lo == 0 {
                return Err(PlanError::InvalidKernelAssumption);
            }
        Ok(())
    }

    pub(super) fn validate_io_shape(
        &self,
        node: &FrozenNode,
        io: TargetIoPlan,
    ) -> Result<(), PlanError> {
        if io.input_storages.len() != node.inputs.len()
            || io.output_storages.len() != node.outputs.len()
        {
            return Err(PlanError::InvalidTargetIoBinding);
        }
        Ok(())
    }

    pub(super) fn validate_node_math_contract(
        &self,
        node: &FrozenNode,
        mode: KernelMathMode,
    ) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < node.outputs.len() {
            let tensor = self
                .find_tensor(node.outputs[i])
                .ok_or(PlanError::MissingTensor)?;
            let allowed = match mode {
                KernelMathMode::Exact => true,
                KernelMathMode::StrictFp32 => matches!(
                    tensor.accuracy,
                    AccuracyContract::StrictFp32
                        | AccuracyContract::AllowAlgebraic
                        | AccuracyContract::AllowFastMath
                        | AccuracyContract::AllowTf32
                        | AccuracyContract::AllowFp16Accumulate
                        | AccuracyContract::AllowBf16Accumulate
                        | AccuracyContract::TargetVerified
                ),
                KernelMathMode::AlgebraicFp => matches!(
                    tensor.accuracy,
                    AccuracyContract::AllowAlgebraic
                        | AccuracyContract::AllowFastMath
                        | AccuracyContract::TargetVerified
                ),
                KernelMathMode::FastMath => matches!(
                    tensor.accuracy,
                    AccuracyContract::AllowFastMath | AccuracyContract::TargetVerified
                ),
                KernelMathMode::Tf32 => matches!(
                    tensor.accuracy,
                    AccuracyContract::AllowTf32
                        | AccuracyContract::AllowFastMath
                        | AccuracyContract::TargetVerified
                ),
                KernelMathMode::Fp16Accumulate => matches!(
                    tensor.accuracy,
                    AccuracyContract::AllowFp16Accumulate
                        | AccuracyContract::AllowFastMath
                        | AccuracyContract::TargetVerified
                ),
                KernelMathMode::Bf16Accumulate => matches!(
                    tensor.accuracy,
                    AccuracyContract::AllowBf16Accumulate
                        | AccuracyContract::AllowFastMath
                        | AccuracyContract::TargetVerified
                ),
                KernelMathMode::QuantizedExactScale => matches!(
                    tensor.accuracy,
                    AccuracyContract::QuantizedExactScale | AccuracyContract::TargetVerified
                ),
                KernelMathMode::TargetVerified => {
                    tensor.accuracy == AccuracyContract::TargetVerified
                }
            };
            if !allowed {
                return Err(PlanError::InvalidAccuracyContract);
            }
            i += 1;
        }
        Ok(())
    }

    fn validate_io_storage_lifetime(
        &self,
        node: &FrozenNode,
        target: ExecutionTarget,
        storage_id: StorageId,
    ) -> Result<(), PlanError> {
        let mut first_step: Option<usize> = None;
        let mut last_step: Option<usize> = None;

        if node.placement == Placement::CpuGpu {
            let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
            let mut i = 0usize;
            while i < mapping.slices.len() {
                if mapping.slices[i].target == target {
                    let slice = u16::try_from(i).map_err(|_| PlanError::InvalidCpuGpuMapping)?;
                    let start = self
                        .cpu_gpu_slice_launch_position(node.id, slice)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    let end = self
                        .cpu_gpu_slice_completion_position(node.id, slice)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    first_step = Some(first_step.map_or(start, |current| current.min(start)));
                    last_step = Some(last_step.map_or(end, |current| current.max(end)));
                }
                i += 1;
            }
        } else {
            first_step = self.node_schedule_start_position(node.id);
            last_step = self.node_schedule_completion_position(node.id);
        }

        let first_step = first_step.ok_or(PlanError::InvalidMemoryLifetime)?;
        let last_step = last_step.ok_or(PlanError::InvalidMemoryLifetime)?;
        let storage = self
            .find_storage(storage_id)
            .ok_or(PlanError::MissingStorage)?;
        let slot = self
            .find_slot(storage.slot)
            .ok_or(PlanError::MissingMemorySlot)?;
        if usize::try_from(slot.live_from_step).unwrap_or(usize::MAX) > first_step
            || usize::try_from(slot.live_until_step).unwrap_or(0) < last_step
        {
            return Err(PlanError::InvalidMemoryLifetime);
        }
        Ok(())
    }

    pub(super) fn validate_workspace_slots(
        &self,
        node: &FrozenNode,
        io: TargetIoPlan,
        slots: &[MemorySlotId],
        gpu_device: Option<u16>,
    ) -> Result<(), PlanError> {
        let target = if gpu_device.is_some() { ExecutionTarget::Gpu } else { ExecutionTarget::Cpu };
        let mut first_step: Option<usize> = None;
        let mut last_step: Option<usize> = None;

        if node.placement == Placement::CpuGpu {
            let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
            let mut i = 0usize;
            while i < mapping.slices.len() {
                if mapping.slices[i].target == target {
                    let slice = u16::try_from(i).map_err(|_| PlanError::InvalidCpuGpuMapping)?;
                    let start = self
                        .cpu_gpu_slice_launch_position(node.id, slice)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    let end = self
                        .cpu_gpu_slice_completion_position(node.id, slice)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    first_step = Some(first_step.map_or(start, |current| current.min(start)));
                    last_step = Some(last_step.map_or(end, |current| current.max(end)));
                }
                i += 1;
            }
        } else {
            first_step = self.node_schedule_start_position(node.id);
            last_step = self.node_schedule_completion_position(node.id);
        }

        let (first_step, last_step) = match (first_step, last_step) {
            (Some(first), Some(last)) => (first, last),
            _ if slots.is_empty() => return Ok(()),
            _ => return Err(PlanError::InvalidWorkspacePlan),
        };

        let mut i = 0usize;
        while i < slots.len() {
            let slot = self.find_slot(slots[i]).ok_or(PlanError::MissingMemorySlot)?;
            if slot.bytes == 0
                || usize::try_from(slot.live_from_step).unwrap_or(usize::MAX) > first_step
                || usize::try_from(slot.live_until_step).unwrap_or(0) < last_step
            {
                return Err(PlanError::InvalidWorkspacePlan);
            }
            let arena = self.find_arena(slot.arena).ok_or(PlanError::MissingArena)?;
            // External arenas are bound through TensorStoragePlan/StorageBinding.
            // A bare workspace slot has no explicit external binding identity,
            // so accepting one would leave resource selection to backend state.
            if arena.class == MemoryClass::External
                || arena.device_allocation == DeviceAllocationPolicy::External
            {
                return Err(PlanError::InvalidWorkspacePlan);
            }
            let accessible = if let Some(device) = gpu_device {
                self.arena_accessible_on_gpu(arena, device)
            } else {
                matches!(
                    arena.class,
                    MemoryClass::HostAligned
                        | MemoryClass::HostPinned
                        | MemoryClass::HostMappedZeroCopy
                        | MemoryClass::HostPageableDirect
                        | MemoryClass::UnifiedManaged
                ) || (arena.class == MemoryClass::External && arena.device_ordinal.is_none())
            };
            if !accessible {
                return Err(PlanError::InvalidWorkspacePlan);
            }

            // Workspace aliasing with node inputs/outputs is forbidden unless
            // the plan representation grows an explicit in-place workspace
            // contract. Implicit overlap would make replay backend-dependent.
            let workspace_end = slot
                .offset_bytes
                .checked_add(slot.bytes)
                .ok_or(PlanError::InvalidWorkspacePlan)?;
            let io_count = io.input_storages.len() + io.output_storages.len();
            let mut binding = 0usize;
            while binding < io_count {
                let storage_id = if binding < io.input_storages.len() {
                    io.input_storages[binding]
                } else {
                    io.output_storages[binding - io.input_storages.len()]
                };
                let storage = self.find_storage(storage_id).ok_or(PlanError::MissingStorage)?;
                let io_slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
                if io_slot.arena == slot.arena && storage.physical_bytes > 0 {
                    let storage_end = io_slot
                        .offset_bytes
                        .checked_add(storage.physical_bytes)
                        .ok_or(PlanError::InvalidWorkspacePlan)?;
                    if slot.offset_bytes < storage_end && io_slot.offset_bytes < workspace_end {
                        return Err(PlanError::InvalidWorkspacePlan);
                    }
                }
                binding += 1;
            }

            if i > 0 && slots[i - 1] >= slots[i] {
                return Err(PlanError::InvalidWorkspacePlan);
            }
            i += 1;
        }
        Ok(())
    }

    fn validate_alias_view_mapping(
        &self,
        node: &FrozenNode,
        io: TargetIoPlan,
    ) -> Result<(), PlanError> {
        if node.inputs.len() != 1
            || node.outputs.len() != 1
            || io.input_storages.len() != 1
            || io.output_storages.len() != 1
        {
            return Err(PlanError::InvalidKernelFamily);
        }
        let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
        let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
        let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
        let dst = self.find_storage(io.output_storages[0]).ok_or(PlanError::MissingStorage)?;
        let expected_external_root = if let Some(root) = src.external_alias_of {
            Some(root)
        } else if src.residency == ResidencyPolicy::ExternalBound {
            Some(src.id)
        } else {
            None
        };
        if src.slot != dst.slot
            || src.physical_bytes != dst.physical_bytes
            || src.format_tag != dst.format_tag
            || src.residency != dst.residency
            || dst.external_alias_of != expected_external_root
            || src.access == StorageAccess::WriteOnly
            || dst.access != StorageAccess::ReadOnly
            || matches!(
                src.physical_layout.kind,
                LayoutKind::Blocked | LayoutKind::Packed | LayoutKind::TargetNative
            )
            || matches!(
                dst.physical_layout.kind,
                LayoutKind::Blocked | LayoutKind::Packed | LayoutKind::TargetNative
            )
        {
            return Err(PlanError::InvalidKernelFamily);
        }

        let element_bytes = i128::try_from(input.dtype.size_bytes())
            .map_err(|_| PlanError::InvalidKernelFamily)?;
        let expected_base = |delta_elements: i128| -> Result<usize, PlanError> {
            let base = i128::try_from(src.base_offset_bytes)
                .map_err(|_| PlanError::InvalidKernelFamily)?;
            let byte_delta = delta_elements
                .checked_mul(element_bytes)
                .ok_or(PlanError::InvalidKernelFamily)?;
            let value = base.checked_add(byte_delta).ok_or(PlanError::InvalidKernelFamily)?;
            if value < 0 {
                return Err(PlanError::InvalidKernelFamily);
            }
            usize::try_from(value).map_err(|_| PlanError::InvalidKernelFamily)
        };

        match node.operation {
            Operation::Reshape => {
                if !src.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !dst.physical_layout.is_row_major_contiguous_for(&output.shape)
                    || src.base_offset_bytes != dst.base_offset_bytes
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            Operation::Transpose { permutation } => {
                if input.shape.rank() != output.shape.rank()
                    || src.base_offset_bytes != dst.base_offset_bytes
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let mut axis = 0usize;
                while axis < output.shape.rank() {
                    let source_axis = usize::from(permutation.axes[axis]);
                    if dst.physical_layout.strides[axis] != src.physical_layout.strides[source_axis] {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    axis += 1;
                }
            }
            Operation::Expand => {
                if dst.physical_layout.kind != LayoutKind::Strided
                    || src.base_offset_bytes != dst.base_offset_bytes
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let pad = output.shape.rank() - input.shape.rank();
                let mut axis = 0usize;
                while axis < output.shape.rank() {
                    let expected_stride = if axis < pad {
                        0
                    } else {
                        let source_axis = axis - pad;
                        let source_dim = input.shape.dim(source_axis).ok_or(PlanError::InvalidKernelFamily)?;
                        let target_dim = output.shape.dim(axis).ok_or(PlanError::InvalidKernelFamily)?;
                        if source_dim == 1 && target_dim != 1 {
                            0
                        } else {
                            src.physical_layout.strides[source_axis]
                        }
                    };
                    if dst.physical_layout.strides[axis] != expected_stride {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    axis += 1;
                }
            }
            Operation::AsStrided(spec) => {
                if output.layout.kind != LayoutKind::Strided
                    || dst.physical_layout.kind != LayoutKind::Strided
                    || src.physical_layout.kind != input.layout.kind
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let mut axis = 0usize;
                while axis < input.shape.rank() {
                    if src.physical_layout.strides[axis] != input.layout.strides[axis] {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    axis += 1;
                }
                axis = 0;
                while axis < output.shape.rank() {
                    if dst.physical_layout.strides[axis] != output.layout.strides[axis] {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    axis += 1;
                }
                if dst.base_offset_bytes != expected_base(spec.storage_offset_elements as i128)? {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            Operation::Narrow(spec) => {
                if input.shape.rank() != output.shape.rank() {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let axis = usize::from(spec.axis);
                let mut d = 0usize;
                while d < input.shape.rank() {
                    if dst.physical_layout.strides[d] != src.physical_layout.strides[d] {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    d += 1;
                }
                let delta = i128::from(spec.start)
                    .checked_mul(src.physical_layout.strides[axis] as i128)
                    .ok_or(PlanError::InvalidKernelFamily)?;
                if dst.base_offset_bytes != expected_base(delta)? {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            Operation::Flip { axes } => {
                if dst.physical_layout.kind != LayoutKind::Strided
                    || input.shape.rank() != output.shape.rank()
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let delta = flipped_alias_delta(
                    &input.shape, axes, &src.physical_layout, &dst.physical_layout,
                )?;
                if dst.base_offset_bytes != expected_base(delta)? {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            Operation::SlidingWindow1d(spec) => {
                if dst.physical_layout.kind != LayoutKind::Strided
                    || src.base_offset_bytes != dst.base_offset_bytes
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let source_stride = src.physical_layout.strides[0];
                let row_stride = source_stride
                    .checked_mul(isize::try_from(spec.step).map_err(|_| PlanError::InvalidKernelFamily)?)
                    .ok_or(PlanError::InvalidKernelFamily)?;
                if dst.physical_layout.strides[0] != row_stride
                    || dst.physical_layout.strides[1] != source_stride
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            _ => return Err(PlanError::InvalidKernelFamily),
        }
        Ok(())
    }

    /// Pure offline-frozen kernel arity ABI check. Semantically valid
    /// one-input Concat/Stack graphs may use an offline-proven generic native
    /// family; fixed 2/4 input families must still enforce their exact arity.
    /// This is NOT a runtime family selector or execution implementation.
    #[inline(always)]
    pub(super) fn generic_join_arity_valid(family: KernelFamily, inputs: usize) -> bool {
        match family {
            KernelFamily::Concat2 | KernelFamily::Stack2 => inputs == 2,
            KernelFamily::Concat4 | KernelFamily::Stack4 => inputs == 4,
            KernelFamily::ConcatGeneric | KernelFamily::StackGeneric => inputs >= 1,
            _ => false,
        }
    }

    pub(super) fn validate_kernel_family(
        &self,
        node: &FrozenNode,
        io: TargetIoPlan,
        family: KernelFamily,
    ) -> Result<(), PlanError> {
        let fused = node.semantic_nodes.len() > 1;
        if fused {
            if matches!(
                family,
                KernelFamily::FusedStaticDataflow
                    | KernelFamily::VendorExact
                    | KernelFamily::TargetNative
            ) {
                return Ok(());
            }
            return Err(PlanError::InvalidKernelFamily);
        }

        if family == KernelFamily::FusedStaticDataflow {
            return Err(PlanError::InvalidKernelFamily);
        }

        let compatible = match family {
            KernelFamily::Generic | KernelFamily::VendorExact | KernelFamily::TargetNative => true,
            KernelFamily::FusedStaticDataflow => false,
            KernelFamily::ContiguousCopy | KernelFamily::StridedCopy => {
                matches!(node.operation, Operation::Copy)
            }
            KernelFamily::AliasView => matches!(
                node.operation,
                Operation::Reshape
                    | Operation::Transpose { .. }
                    | Operation::Expand
                    | Operation::AsStrided(_)
                    | Operation::Narrow(_)
                    | Operation::Flip { .. }
                    | Operation::SlidingWindow1d(_)
            ),
            KernelFamily::ExpandMaterialize => matches!(node.operation, Operation::Expand),
            KernelFamily::FlipContiguous2d | KernelFamily::FlipGeneric => {
                matches!(node.operation, Operation::Flip { .. })
            }
            KernelFamily::Concat2 | KernelFamily::Concat4 | KernelFamily::ConcatGeneric => {
                matches!(node.operation, Operation::Concat { .. })
            }
            KernelFamily::Stack2 | KernelFamily::Stack4 | KernelFamily::StackGeneric => {
                matches!(node.operation, Operation::Stack { .. })
            }
            KernelFamily::PadConstantContiguous | KernelFamily::PadMapped => {
                matches!(node.operation, Operation::Pad(_))
            }
            KernelFamily::ResampleNearest
            | KernelFamily::ResampleBilinear
            | KernelFamily::DownsampleMeanScale2
            | KernelFamily::DownsampleMeanGeneric
            | KernelFamily::DownsampleMeanIntegral2d => {
                matches!(node.operation, Operation::Resample2d(_))
            }
            KernelFamily::SlidingWindowContiguous => {
                matches!(node.operation, Operation::SlidingWindow1d(_))
            }
            KernelFamily::ZeroInsert1d => {
                matches!(node.operation, Operation::ZeroInsert1d { .. })
            }
            KernelFamily::TileMaterialize | KernelFamily::RepeatSingletonLast => {
                matches!(node.operation, Operation::Tile(_))
            }
            KernelFamily::RepeatInterleaveLast | KernelFamily::RepeatInterleaveGeneric => {
                matches!(node.operation, Operation::RepeatInterleave(_))
            }
            KernelFamily::ContiguousGather
            | KernelFamily::ChannelsLastGather4d
            | KernelFamily::ChannelsLastGather5d
            | KernelFamily::FancyGatherOneAxis => matches!(node.operation, Operation::Gather { .. }),
            KernelFamily::MetadataAssignTensor | KernelFamily::MetadataAssignScalar => {
                matches!(node.operation, Operation::Scatter(_))
            }
            KernelFamily::Elementwise
            | KernelFamily::ContiguousElementwise
            | KernelFamily::BroadcastElementwise
            | KernelFamily::MaskElementwise
            | KernelFamily::ChannelwisePRelu => {
                matches!(node.operation, Operation::Elementwise { .. })
            }
            KernelFamily::ScalarElementwise | KernelFamily::ScalarPowerThreeHalves => {
                matches!(node.operation, Operation::ScalarElementwise(_))
            }
            KernelFamily::Select => matches!(node.operation, Operation::Select { .. }),
            KernelFamily::RowwiseReduction
            | KernelFamily::TreeReduction
            | KernelFamily::TwoPassReduction => {
                matches!(node.operation, Operation::Reduction { .. })
            }
            KernelFamily::RowwiseNormalization | KernelFamily::FusedNormalization => {
                matches!(node.operation, Operation::Normalization(_))
            }
            KernelFamily::FusedSoftmax => {
                matches!(node.operation, Operation::Softmax { log: false, .. })
            }
            KernelFamily::FusedLogSoftmax => {
                matches!(node.operation, Operation::Softmax { log: true, .. })
            }
            KernelFamily::ParallelPrefixScan | KernelFamily::SegmentedScan => {
                matches!(node.operation, Operation::Scan(_))
            }
            KernelFamily::BitonicSort | KernelFamily::ComparisonSort => {
                matches!(node.operation, Operation::Sort(_))
            }
            KernelFamily::TopKSelection | KernelFamily::TopKSortTrim => {
                matches!(node.operation, Operation::TopK(_))
            }
            KernelFamily::FftSmallLastDimGpu
            | KernelFamily::FftRadix2
            | KernelFamily::FftBluestein => {
                matches!(node.operation, Operation::FourierTransform(_))
            }
            KernelFamily::CpuLinalgSmallStatic | KernelFamily::CpuLinalgBatched => {
                matches!(node.operation, Operation::DenseLinearAlgebra(_))
            }
            KernelFamily::CpuDotSimd
            | KernelFamily::CpuVecMatSimd
            | KernelFamily::CpuVecMatAxpy
            | KernelFamily::CpuMatVecSimd
            | KernelFamily::CpuSgemmFixedTiny
            | KernelFamily::CpuSgemmBatchedSmall
            | KernelFamily::CpuSgemmDirect
            | KernelFamily::CpuSgemmDirectAxpy
            | KernelFamily::CpuSgemmPacked4x8
            | KernelFamily::CpuSgemmPacked6xSimd
            | KernelFamily::CpuSgemmPrepackedB4x8 => matches!(node.operation, Operation::MatMul { .. }),
        };
        if !compatible {
            return Err(PlanError::InvalidKernelFamily);
        }

        match family {
            KernelFamily::AliasView => {
                if node.placement == Placement::CpuGpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
                self.validate_alias_view_mapping(node, io)?;
            }
            KernelFamily::Concat2 | KernelFamily::Concat4 | KernelFamily::ConcatGeneric
            | KernelFamily::Stack2 | KernelFamily::Stack4 | KernelFamily::StackGeneric => {
                if !Self::generic_join_arity_valid(family, node.inputs.len()) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::PadConstantContiguous => {
                if !matches!(node.operation, Operation::Pad(spec) if spec.mode == PadMode::Constant) {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
                let dst = self.find_storage(io.output_storages[0]).ok_or(PlanError::MissingStorage)?;
                if !src.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !dst.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ResampleNearest => {
                if !matches!(node.operation, Operation::Resample2d(spec) if spec.mode == ResampleMode::Nearest) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ResampleBilinear => {
                if !matches!(node.operation, Operation::Resample2d(spec) if spec.mode == ResampleMode::Bilinear) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::DownsampleMeanScale2 => {
                if !matches!(node.operation, Operation::Resample2d(spec) if spec.mode == ResampleMode::MeanDownsample && spec.scale == 2) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::DownsampleMeanGeneric => {
                if !matches!(node.operation, Operation::Resample2d(spec) if spec.mode == ResampleMode::MeanDownsample) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::DownsampleMeanIntegral2d => {
                if node.placement != Placement::Cpu
                    || !matches!(node.operation, Operation::Resample2d(spec) if spec.mode == ResampleMode::MeanDownsample && spec.scale > 1)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
                let dst = self.find_storage(io.output_storages[0]).ok_or(PlanError::MissingStorage)?;
                if input.shape.rank() != 2
                    || !src.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !dst.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::SlidingWindowContiguous => {
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
                if !src.physical_layout.is_row_major_contiguous_for(&input.shape) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ZeroInsert1d => {
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
                let dst = self.find_storage(io.output_storages[0]).ok_or(PlanError::MissingStorage)?;
                if !matches!(node.operation, Operation::ZeroInsert1d { factor } if factor > 1)
                    || !src.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !dst.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::RepeatSingletonLast => {
                let Operation::Tile(spec) = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let rank = input.shape.rank();
                if rank == 0
                    || input.shape.dim(rank - 1) != Some(1)
                    || spec.repeats[rank - 1] <= 1
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let mut d = 0usize;
                while d + 1 < rank {
                    if spec.repeats[d] != 1 {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    d += 1;
                }
            }
            KernelFamily::RepeatInterleaveLast => {
                let Operation::RepeatInterleave(spec) = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let src = self.find_storage(io.input_storages[0]).ok_or(PlanError::MissingStorage)?;
                let dst = self.find_storage(io.output_storages[0]).ok_or(PlanError::MissingStorage)?;
                let rank = input.shape.rank();
                if rank == 0
                    || usize::from(spec.axis) + 1 != rank
                    || !src.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !dst.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::TileMaterialize | KernelFamily::RepeatInterleaveGeneric => {}
            KernelFamily::ContiguousElementwise => {
                let mut i = 0usize;
                while i < node.inputs.len() {
                    let tensor = self.find_tensor(node.inputs[i]).ok_or(PlanError::MissingTensor)?;
                    let storage = self
                        .find_storage(io.input_storages[i])
                        .ok_or(PlanError::MissingStorage)?;
                    if !storage.physical_layout.is_row_major_contiguous_for(&tensor.shape) {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    i += 1;
                }
                i = 0;
                while i < node.outputs.len() {
                    let tensor = self.find_tensor(node.outputs[i]).ok_or(PlanError::MissingTensor)?;
                    let storage = self
                        .find_storage(io.output_storages[i])
                        .ok_or(PlanError::MissingStorage)?;
                    if !storage.physical_layout.is_row_major_contiguous_for(&tensor.shape) {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    i += 1;
                }
                if let Operation::Elementwise { broadcast_signature, .. } = node.operation {
                    if broadcast_signature != 0 {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                }
            }
            KernelFamily::BroadcastElementwise => {
                if let Operation::Elementwise { broadcast_signature, .. } = node.operation {
                    if broadcast_signature == 0 {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                }
            }
            KernelFamily::MaskElementwise => {
                if !matches!(
                    node.operation,
                    Operation::Elementwise {
                        kind: ElementwiseKind::LogicalAnd
                            | ElementwiseKind::LogicalOr
                            | ElementwiseKind::LogicalNot,
                        ..
                    }
                ) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ChannelwisePRelu => {
                if !matches!(
                    node.operation,
                    Operation::Elementwise { kind: ElementwiseKind::PRelu, .. }
                ) || node.inputs.len() != 2
                    || node.outputs.len() != 1
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let alpha = self.find_tensor(node.inputs[1]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let rank = output.shape.rank();
                if rank == 0
                    || input.shape != output.shape
                    || alpha.shape.rank() != 1
                    || alpha.shape.dim(0) != output.shape.dim(rank - 1)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::TopKSelection => {
                if !matches!(node.operation, Operation::TopK(spec) if spec.k <= 32) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ScalarPowerThreeHalves => {
                let compatible = match node.operation {
                    Operation::ScalarElementwise(spec) => {
                        spec.kind == ScalarElementwiseKind::Power
                            && spec.scalar0_f32_bits == 1.5f32.to_bits()
                    }
                    _ => false,
                };
                if !compatible {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::CpuDotSimd
            | KernelFamily::CpuVecMatSimd
            | KernelFamily::CpuVecMatAxpy
            | KernelFamily::CpuMatVecSimd => {
                if node.placement != Placement::Cpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let Operation::MatMul { transpose_lhs, transpose_rhs } = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                if transpose_lhs || transpose_rhs || node.inputs.len() != 2 || node.outputs.len() != 1 {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let lhs = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let rhs = self.find_tensor(node.inputs[1]).ok_or(PlanError::MissingTensor)?;
                let out = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let rhs_storage = self
                    .find_storage(io.input_storages[1])
                    .ok_or(PlanError::MissingStorage)?;
                let out_storage = self
                    .find_storage(io.output_storages[0])
                    .ok_or(PlanError::MissingStorage)?;

                match family {
                    KernelFamily::CpuDotSimd => {
                        if lhs.shape.rank() != 1
                            || rhs.shape.rank() != 1
                            || out.shape.rank() != 0
                            || lhs.shape.dim(0) != rhs.shape.dim(0)
                            || lhs.shape.dim(0) == Some(0)
                        {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                    }
                    KernelFamily::CpuVecMatSimd => {
                        if lhs.shape.rank() != 1 || rhs.shape.rank() != 2 || out.shape.rank() != 1 {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        let k = lhs.shape.dim(0).ok_or(PlanError::InvalidKernelFamily)?;
                        if k == 0
                            || rhs.shape.dim(0) != Some(k)
                            || rhs.shape.dim(1) == Some(0)
                            || out.shape.dim(0) != rhs.shape.dim(1)
                        {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        if out_storage.physical_layout.strides[0] != 1 {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                    }
                    KernelFamily::CpuVecMatAxpy => {
                        if lhs.shape.rank() != 1 || rhs.shape.rank() != 2 || out.shape.rank() != 1 {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        let k = lhs.shape.dim(0).ok_or(PlanError::InvalidKernelFamily)?;
                        if k == 0
                            || rhs.shape.dim(0) != Some(k)
                            || rhs.shape.dim(1) == Some(0)
                            || out.shape.dim(0) != rhs.shape.dim(1)
                        {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        // The supplied fast path performs out += a[p] * B[p, :] and
                        // therefore requires rows of B and the output vector contiguous.
                        if rhs_storage.physical_layout.strides[1] != 1
                            || out_storage.physical_layout.strides[0] != 1
                        {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                    }
                    KernelFamily::CpuMatVecSimd => {
                        if lhs.shape.rank() != 2 || rhs.shape.rank() != 1 || out.shape.rank() != 1 {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        let k = lhs.shape.dim(1).ok_or(PlanError::InvalidKernelFamily)?;
                        if k == 0
                            || lhs.shape.dim(0) == Some(0)
                            || rhs.shape.dim(0) != Some(k)
                            || out.shape.dim(0) != lhs.shape.dim(0)
                        {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        if out_storage.physical_layout.strides[0] != 1 {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                    }
                    _ => return Err(PlanError::InvalidKernelFamily),
                }

            }
            KernelFamily::CpuSgemmFixedTiny => {
                if node.placement != Placement::Cpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let Operation::MatMul { transpose_lhs, transpose_rhs } = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                if transpose_lhs || transpose_rhs {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let lhs = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let rhs = self.find_tensor(node.inputs[1]).ok_or(PlanError::MissingTensor)?;
                let out = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                if lhs.shape.rank() != 2 || rhs.shape.rank() != 2 || out.shape.rank() != 2 {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let m = lhs.shape.dim(0).ok_or(PlanError::InvalidKernelFamily)?;
                let k = lhs.shape.dim(1).ok_or(PlanError::InvalidKernelFamily)?;
                let kb = rhs.shape.dim(0).ok_or(PlanError::InvalidKernelFamily)?;
                let n = rhs.shape.dim(1).ok_or(PlanError::InvalidKernelFamily)?;
                let is_2 = m == 2 && k == 2 && kb == 2 && n == 2;
                let is_3 = m == 3 && k == 3 && kb == 3 && n == 3;
                if !(is_2 || is_3) || out.shape.dim(0) != Some(m) || out.shape.dim(1) != Some(n) {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::CpuSgemmBatchedSmall => {
                if node.placement != Placement::Cpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let Operation::MatMul { transpose_lhs, transpose_rhs } = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                if transpose_lhs || transpose_rhs {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let lhs = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let rhs = self.find_tensor(node.inputs[1]).ok_or(PlanError::MissingTensor)?;
                let out = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let rank = lhs.shape.rank();
                if rank < 3 || rhs.shape.rank() != rank || out.shape.rank() != rank {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let m = lhs.shape.dim(rank - 2).ok_or(PlanError::InvalidKernelFamily)?;
                let k = lhs.shape.dim(rank - 1).ok_or(PlanError::InvalidKernelFamily)?;
                let kb = rhs.shape.dim(rank - 2).ok_or(PlanError::InvalidKernelFamily)?;
                let n = rhs.shape.dim(rank - 1).ok_or(PlanError::InvalidKernelFamily)?;
                if k != kb || m == 0 || k == 0 || n == 0 || m > 64 || k > 64 || n > 64
                    || out.shape.dim(rank - 2) != Some(m)
                    || out.shape.dim(rank - 1) != Some(n)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let mut d = 0usize;
                while d + 2 < rank {
                    let batch_dim = lhs.shape.dim(d).ok_or(PlanError::InvalidKernelFamily)?;
                    if rhs.shape.dim(d) != Some(batch_dim) || out.shape.dim(d) != Some(batch_dim) {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    d += 1;
                }
                let rhs_storage = self
                    .find_storage(io.input_storages[1])
                    .ok_or(PlanError::MissingStorage)?;
                let out_storage = self
                    .find_storage(io.output_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                if rhs_storage.physical_layout.strides[rank - 1] != 1
                    || out_storage.physical_layout.strides[rank - 1] != 1
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::FftSmallLastDimGpu
            | KernelFamily::FftRadix2
            | KernelFamily::FftBluestein => {
                let Operation::FourierTransform(spec) = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                let reference_tensor = match spec.kind {
                    FourierTransformKind::RealInverse => {
                        self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?
                    }
                    _ => self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?,
                };
                let axis = usize::from(spec.axis);
                let transform_len = reference_tensor
                    .shape
                    .dim(axis)
                    .ok_or(PlanError::InvalidKernelFamily)?;
                let is_pow2 = transform_len > 0 && transform_len.is_power_of_two();
                if family == KernelFamily::FftSmallLastDimGpu {
                    if node.placement != Placement::Gpu
                        || axis + 1 != reference_tensor.shape.rank()
                        || !(transform_len == 2 || transform_len == 4)
                    {
                        return Err(PlanError::InvalidKernelFamily);
                    }
                    let mut i = 0usize;
                    while i < node.inputs.len() {
                        let tensor = self.find_tensor(node.inputs[i]).ok_or(PlanError::MissingTensor)?;
                        let storage = self
                            .find_storage(io.input_storages[i])
                            .ok_or(PlanError::MissingStorage)?;
                        if !storage.physical_layout.is_row_major_contiguous_for(&tensor.shape) {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        i += 1;
                    }
                    i = 0;
                    while i < node.outputs.len() {
                        let tensor = self.find_tensor(node.outputs[i]).ok_or(PlanError::MissingTensor)?;
                        let storage = self
                            .find_storage(io.output_storages[i])
                            .ok_or(PlanError::MissingStorage)?;
                        if !storage.physical_layout.is_row_major_contiguous_for(&tensor.shape) {
                            return Err(PlanError::InvalidKernelFamily);
                        }
                        i += 1;
                    }
                } else if (family == KernelFamily::FftRadix2) != is_pow2 {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::CpuLinalgSmallStatic => {
                if node.placement != Placement::Cpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let Operation::DenseLinearAlgebra(spec) = node.operation else {
                    return Err(PlanError::InvalidKernelFamily);
                };
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let rank = input.shape.rank();
                let n = input.shape.dim(rank - 1).ok_or(PlanError::InvalidKernelFamily)?;
                let batch = if rank <= 2 {
                    1usize
                } else {
                    let mut product = 1usize;
                    let mut d = 0usize;
                    while d + 2 < rank {
                        product = product
                            .checked_mul(input.shape.dim(d).ok_or(PlanError::InvalidKernelFamily)?)
                            .ok_or(PlanError::InvalidKernelFamily)?;
                        d += 1;
                    }
                    product
                };
                if batch != 1
                    || !(n == 2 || n == 3)
                    || !matches!(
                        spec.kind,
                        DenseLinalgKind::Solve | DenseLinalgKind::Inverse | DenseLinalgKind::CholeskyLower
                    )
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::CpuLinalgBatched => {
                if node.placement != Placement::Cpu {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ContiguousCopy => {
                if node.inputs.len() != 1
                    || node.outputs.len() != 1
                    || io.input_storages.len() != 1
                    || io.output_storages.len() != 1
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let input_storage = self
                    .find_storage(io.input_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                let output_storage = self
                    .find_storage(io.output_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                if !input_storage.physical_layout.is_row_major_contiguous_for(&input.shape)
                    || !output_storage.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ContiguousGather => {
                if node.inputs.len() != 2
                    || node.outputs.len() != 1
                    || io.input_storages.len() != 2
                    || io.output_storages.len() != 1
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let source = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let output = self.find_tensor(node.outputs[0]).ok_or(PlanError::MissingTensor)?;
                let source_storage = self
                    .find_storage(io.input_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                let output_storage = self
                    .find_storage(io.output_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                if !source_storage.physical_layout.is_row_major_contiguous_for(&source.shape)
                    || !output_storage.physical_layout.is_row_major_contiguous_for(&output.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::ChannelsLastGather4d | KernelFamily::ChannelsLastGather5d => {
                if node.inputs.len() < 2 || io.input_storages.len() < 2 {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
                let storage = self
                    .find_storage(io.input_storages[0])
                    .ok_or(PlanError::MissingStorage)?;
                let expected_rank = if family == KernelFamily::ChannelsLastGather4d { 4 } else { 5 };
                if input.shape.rank() != expected_rank
                    || !storage.physical_layout.is_channels_last_contiguous_for(&input.shape)
                {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            KernelFamily::MetadataAssignScalar | KernelFamily::MetadataAssignTensor => {
                if node.inputs.len() != 3 {
                    return Err(PlanError::InvalidKernelFamily);
                }
                let values = self.find_tensor(node.inputs[2]).ok_or(PlanError::MissingTensor)?;
                let scalar = values.shape.rank() == 0;
                if (family == KernelFamily::MetadataAssignScalar) != scalar {
                    return Err(PlanError::InvalidKernelFamily);
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn validate_cpu_io(
        &self,
        node: &FrozenNode,
        cpu: &CpuKernelPlan,
    ) -> Result<(), PlanError> {
        self.validate_io_shape(node, cpu.io)?;
        self.validate_kernel_assumptions(cpu.io, cpu.assumptions)?;
        if cpu.assumptions.unchecked_indexing
            && matches!(node.operation, Operation::Gather { .. } | Operation::Scatter(_))
        {
            return Err(PlanError::InvalidKernelAssumption);
        }
        self.validate_node_math_contract(node, cpu.math_mode)?;
        self.validate_kernel_family(node, cpu.io, cpu.family)?;
        self.validate_workspace_slots(node, cpu.io, cpu.workspace_slots, None)?;
        if cpu.family == KernelFamily::DownsampleMeanIntegral2d {
            let input = self.find_tensor(node.inputs[0]).ok_or(PlanError::MissingTensor)?;
            let h = input.shape.dim(0).ok_or(PlanError::InvalidWorkspacePlan)?;
            let w = input.shape.dim(1).ok_or(PlanError::InvalidWorkspacePlan)?;
            let required_bytes = h
                .checked_add(1)
                .and_then(|rows| w.checked_add(1).and_then(|cols| rows.checked_mul(cols)))
                .and_then(|elements| elements.checked_mul(DType::F32.size_bytes()))
                .ok_or(PlanError::InvalidWorkspacePlan)?;
            let mut found_prefix_workspace = false;
            let mut workspace = 0usize;
            while workspace < cpu.workspace_slots.len() {
                let slot = self
                    .find_slot(cpu.workspace_slots[workspace])
                    .ok_or(PlanError::MissingMemorySlot)?;
                if slot.bytes >= required_bytes {
                    found_prefix_workspace = true;
                    break;
                }
                workspace += 1;
            }
            if !found_prefix_workspace {
                return Err(PlanError::InvalidWorkspacePlan);
            }
        }
        if cpu.persistent_packed_weights {
            if !matches!(node.operation, Operation::MatMul { .. })
                || !cpu.pack_b
                || node.inputs.len() < 2
            {
                return Err(PlanError::InvalidWorkspacePlan);
            }
            let weight = self.find_tensor(node.inputs[1]).ok_or(PlanError::MissingTensor)?;
            if weight.reuse != ReusePattern::PersistentWeight
                || self.tensor_producer(weight.id).is_some()
            {
                // A persistent packed weight is immutable replay input. If the
                // same replay produces it, persistence across replays is ambiguous.
                return Err(PlanError::InvalidWorkspacePlan);
            }
            let required_last = self.schedule.len().saturating_sub(1);
            let mut persistent_slot = false;
            let mut w = 0usize;
            while w < cpu.workspace_slots.len() {
                let slot = self
                    .find_slot(cpu.workspace_slots[w])
                    .ok_or(PlanError::MissingMemorySlot)?;
                if slot.lifetime == LifetimeClass::Persistent
                    && slot.live_from_step == 0
                    && usize::try_from(slot.live_until_step).unwrap_or(0) == required_last
                {
                    persistent_slot = true;
                    break;
                }
                w += 1;
            }
            if !persistent_slot {
                return Err(PlanError::InvalidWorkspacePlan);
            }
        }
        let mut i = 0usize;
        while i < node.inputs.len() {
            let storage = self
                .find_storage(cpu.io.input_storages[i])
                .ok_or(PlanError::MissingStorage)?;
            if storage.tensor != node.inputs[i] {
                return Err(PlanError::StorageTensorMismatch);
            }
            if storage.access == StorageAccess::WriteOnly {
                return Err(PlanError::InvalidStorageAccess);
            }
            let arena = self
                .arena_of_storage(storage.id)
                .ok_or(PlanError::MissingArena)?;
            let accessible = match arena.class {
                MemoryClass::HostAligned
                | MemoryClass::HostPinned
                | MemoryClass::HostMappedZeroCopy
                | MemoryClass::HostPageableDirect
                | MemoryClass::UnifiedManaged => true,
                MemoryClass::DeviceLocal => false,
                MemoryClass::External => arena.device_ordinal.is_none(),
            };
            if !accessible {
                return Err(PlanError::InvalidTargetIoBinding);
            }
            self.validate_io_storage_lifetime(node, ExecutionTarget::Cpu, storage.id)?;
            i += 1;
        }

        i = 0;
        while i < node.outputs.len() {
            let storage = self
                .find_storage(cpu.io.output_storages[i])
                .ok_or(PlanError::MissingStorage)?;
            if storage.tensor != node.outputs[i] {
                return Err(PlanError::StorageTensorMismatch);
            }
            if cpu.family == KernelFamily::AliasView {
                if storage.access != StorageAccess::ReadOnly {
                    return Err(PlanError::InvalidStorageAccess);
                }
            } else if storage.access == StorageAccess::ReadOnly {
                return Err(PlanError::InvalidStorageAccess);
            }
            let arena = self
                .arena_of_storage(storage.id)
                .ok_or(PlanError::MissingArena)?;
            let accessible = match arena.class {
                MemoryClass::HostAligned
                | MemoryClass::HostPinned
                | MemoryClass::HostMappedZeroCopy
                | MemoryClass::HostPageableDirect
                | MemoryClass::UnifiedManaged => true,
                MemoryClass::DeviceLocal => false,
                MemoryClass::External => arena.device_ordinal.is_none(),
            };
            if !accessible {
                return Err(PlanError::InvalidTargetIoBinding);
            }
            self.validate_io_storage_lifetime(node, ExecutionTarget::Cpu, storage.id)?;
            i += 1;
        }
        Ok(())
    }

    pub(super) fn validate_gpu_io(
        &self,
        node: &FrozenNode,
        gpu: &GpuKernelPlan,
    ) -> Result<(), PlanError> {
        self.validate_io_shape(node, gpu.io)?;
        self.validate_kernel_assumptions(gpu.io, gpu.assumptions)?;
        if gpu.assumptions.unchecked_indexing
            && matches!(node.operation, Operation::Gather { .. } | Operation::Scatter(_))
        {
            return Err(PlanError::InvalidKernelAssumption);
        }
        self.validate_node_math_contract(node, gpu.math_mode)?;
        self.validate_kernel_family(node, gpu.io, gpu.family)?;
        self.validate_workspace_slots(node, gpu.io, gpu.workspace_slots, Some(gpu.device_ordinal))?;
        let validate_storage = |storage_id: StorageId, tensor_id: TensorId, write: bool|
            -> Result<(), PlanError> {
            let storage = self.find_storage(storage_id).ok_or(PlanError::MissingStorage)?;
            if storage.tensor != tensor_id {
                return Err(PlanError::StorageTensorMismatch);
            }
            if (write && storage.access == StorageAccess::ReadOnly)
                || (!write && storage.access == StorageAccess::WriteOnly)
            {
                return Err(PlanError::InvalidStorageAccess);
            }
            let arena = self
                .arena_of_storage(storage.id)
                .ok_or(PlanError::MissingArena)?;
            let accessible = self.arena_accessible_on_gpu(arena, gpu.device_ordinal);
            if !accessible {
                return Err(PlanError::InvalidTargetIoBinding);
            }
            self.validate_io_storage_lifetime(node, ExecutionTarget::Gpu, storage.id)?;
            Ok(())
        };

        let mut i = 0usize;
        while i < node.inputs.len() {
            validate_storage(gpu.io.input_storages[i], node.inputs[i], false)?;
            i += 1;
        }
        i = 0;
        while i < node.outputs.len() {
            validate_storage(
                gpu.io.output_storages[i],
                node.outputs[i],
                gpu.family != KernelFamily::AliasView,
            )?;
            i += 1;
        }

        if matches!(
            gpu.matrix_acceleration,
            MatrixAcceleration::SparseMma
                | MatrixAcceleration::SparseTcgen05
                | MatrixAcceleration::SparseBlockScaledMma
        ) {
            let mut sparse_input = false;
            i = 0;
            while i < node.inputs.len() {
                let tensor = self.find_tensor(node.inputs[i]).ok_or(PlanError::MissingTensor)?;
                if matches!(
                    tensor.sparsity.pattern,
                    SparsityPattern::Structured1To2
                        | SparsityPattern::Structured2To4
                        | SparsityPattern::Structured4To8
                ) {
                    sparse_input = true;
                    break;
                }
                i += 1;
            }
            if !sparse_input {
                return Err(PlanError::InvalidGpuTuning);
            }
        }
        Ok(())
    }

    pub(super) fn validate_artifact_ref(&self, artifact: ArtifactRef) -> Result<(), PlanError> {
        if artifact.artifact.is_empty()
            || artifact.entry.is_empty()
            || (artifact.artifact_hash_hi == 0 && artifact.artifact_hash_lo == 0)
            || (artifact.target_profile_hash_hi == 0 && artifact.target_profile_hash_lo == 0)
            || artifact.target_profile_hash_hi != self.hardware.profile_hash_hi
            || artifact.target_profile_hash_lo != self.hardware.profile_hash_lo
        {
            return Err(PlanError::InvalidArtifact);
        }
        Ok(())
    }

    pub(super) fn validate_library_algorithm(
        &self,
        artifact: ArtifactKind,
        algorithm: Option<LibraryAlgorithmPlan>,
    ) -> Result<(), PlanError> {
        match (artifact, algorithm) {
            (ArtifactKind::LibraryKernel, Some(algo)) => {
                if algo.provider == LibraryProvider::None
                    || algo.provider_version.is_empty()
                    || (algo.configuration_hash_hi == 0 && algo.configuration_hash_lo == 0)
                    || !algo.exact_algorithm_binding
                    || !algo.provider_threading_fixed
                    || !algo.runtime_heuristics_disabled
                    || !algo.runtime_autotune_disabled
                {
                    return Err(PlanError::InvalidLibraryAlgorithm);
                }
                let mut i = 1usize;
                while i < algo.knobs.len() {
                    if algo.knobs[i - 1].key >= algo.knobs[i].key {
                        return Err(PlanError::InvalidLibraryAlgorithm);
                    }
                    i += 1;
                }
                match (algo.workspace_bytes, algo.workspace_slot) {
                    (0, None) => {}
                    (0, Some(_)) => return Err(PlanError::InvalidLibraryAlgorithm),
                    (_, None) => return Err(PlanError::InvalidLibraryAlgorithm),
                    (bytes, Some(slot_id)) => {
                        let slot = self.find_slot(slot_id).ok_or(PlanError::MissingMemorySlot)?;
                        if slot.bytes < bytes {
                            return Err(PlanError::StorageTooSmall);
                        }
                    }
                }
            }
            (ArtifactKind::LibraryKernel, None) => {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            (_, Some(_)) => return Err(PlanError::InvalidLibraryAlgorithm),
            (_, None) => {}
        }
        Ok(())
    }

}

// =============================================================================
// Cpu Mapping validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_cpu_mapping(&self, node: &FrozenNode, cpu: &CpuKernelPlan) -> Result<(), PlanError> {
        self.validate_artifact_ref(cpu.artifact)?;
        self.validate_library_algorithm(cpu.artifact.kind, cpu.library_algorithm)?;
        if let Some(algorithm) = cpu.library_algorithm {
            if !matches!(
                algorithm.provider,
                LibraryProvider::OneDnn
                    | LibraryProvider::Mkl
                    | LibraryProvider::OpenBlas
                    | LibraryProvider::Accelerate
                    | LibraryProvider::Blis
                    | LibraryProvider::Blas
                    | LibraryProvider::TargetNative
            ) || algorithm.provider_thread_count == 0
            {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            let logical_cpu_count = self
                .hardware
                .cpu
                .map(|cpu_hw| cpu_hw.processors.len())
                .unwrap_or(0);
            if usize::from(algorithm.provider_thread_count) > logical_cpu_count
                || algorithm.provider_thread_bindings.len()
                    != usize::from(algorithm.provider_thread_count)
                || cpu.parallel_kind != CpuParallelKind::SingleThread
            {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            let mut thread_index = 0usize;
            while thread_index < algorithm.provider_thread_bindings.len() {
                let binding = algorithm.provider_thread_bindings[thread_index];
                let processor = self
                    .find_cpu_processor(binding.processor_group, binding.logical_cpu_id)
                    .ok_or(PlanError::InvalidLibraryAlgorithm)?;
                if processor.numa_node != binding.numa_node {
                    return Err(PlanError::InvalidLibraryAlgorithm);
                }
                let mut other = thread_index + 1;
                while other < algorithm.provider_thread_bindings.len() {
                    let rhs = algorithm.provider_thread_bindings[other];
                    if binding.processor_group == rhs.processor_group
                        && binding.logical_cpu_id == rhs.logical_cpu_id
                    {
                        return Err(PlanError::InvalidLibraryAlgorithm);
                    }
                    other += 1;
                }
                thread_index += 1;
            }
            let provider_matches_operation = match algorithm.provider {
                LibraryProvider::OneDnn => matches!(
                    node.operation,
                    Operation::MatMul { .. }
                        | Operation::TensorContract { .. }
                        | Operation::Convolution(_)
                        | Operation::Attention(_)
                        | Operation::Normalization(_)
                        | Operation::Pooling(_)
                        | Operation::Softmax { .. }
                ),
                LibraryProvider::Mkl | LibraryProvider::Accelerate => matches!(
                    node.operation,
                    Operation::MatMul { .. }
                        | Operation::TensorContract { .. }
                        | Operation::DenseLinearAlgebra(_)
                        | Operation::FourierTransform(_)
                ),
                LibraryProvider::OpenBlas | LibraryProvider::Blis | LibraryProvider::Blas => matches!(
                    node.operation,
                    Operation::MatMul { .. }
                        | Operation::TensorContract { .. }
                        | Operation::DenseLinearAlgebra(_)
                ),
                LibraryProvider::TargetNative => true,
                _ => false,
            };
            if !provider_matches_operation {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            if let Some(workspace) = algorithm.workspace_slot {
                if !cpu.workspace_slots.contains(&workspace) {
                    return Err(PlanError::InvalidWorkspacePlan);
                }
            }
        }
        if (cpu.artifact.kind == ArtifactKind::LibraryKernel) != (cpu.family == KernelFamily::VendorExact) {
            return Err(PlanError::InvalidKernelFamily);
        }

        match (cpu.artifact.kind, cpu.rust_codegen) {
            (ArtifactKind::LibraryKernel, None) => {}
            (
                ArtifactKind::RustAotFunction
                | ArtifactKind::NativeObject
                | ArtifactKind::StaticLibrary,
                Some(codegen),
            ) => {
                if codegen.rustc_version.is_empty()
                    || codegen.target_triple.is_empty()
                    || codegen.target_cpu.is_empty()
                    || codegen.opt_level != 3
                    || codegen.codegen_units == 0
                    || codegen.panic_strategy == RustPanicStrategy::Unwind
                    || (codegen.panic_strategy == RustPanicStrategy::ImmediateAbort
                        && !codegen.nightly_codegen)
                    || (codegen.pgo == PgoMode::SampleProfileUse && !codegen.nightly_codegen)
                    || codegen.overflow_checks
                    || codegen.debug_assertions
                    || codegen.incremental
                    || (!codegen.tune_cpu.is_empty() && !codegen.nightly_codegen)
                    || (matches!(
                        codegen.source,
                        CpuKernelSource::PortableSimdNightly
                            | CpuKernelSource::CoreIntrinsicsNightly
                    ) && !codegen.nightly_codegen)
                    || (codegen.source == CpuKernelSource::AutoVectorizedRust
                        && (!codegen.vectorize_loops || !codegen.vectorize_slp))
                {
                    return Err(PlanError::InvalidRustCodegen);
                }
                if codegen.float_mode == RustFloatMode::Algebraic
                    && !matches!(
                        cpu.math_mode,
                        KernelMathMode::AlgebraicFp
                            | KernelMathMode::FastMath
                            | KernelMathMode::TargetVerified
                    )
                {
                    return Err(PlanError::InvalidRustCodegen);
                }
                if cpu.math_mode == KernelMathMode::AlgebraicFp
                    && codegen.float_mode != RustFloatMode::Algebraic
                {
                    return Err(PlanError::InvalidRustCodegen);
                }
                if codegen.pgo != PgoMode::None
                    && codegen.pgo_profile_hash_hi == 0
                    && codegen.pgo_profile_hash_lo == 0
                {
                    return Err(PlanError::InvalidRustCodegen);
                }
                if codegen.post_link_optimization != PostLinkOptimization::None
                    && codegen.post_link_profile_hash_hi == 0
                    && codegen.post_link_profile_hash_lo == 0
                {
                    return Err(PlanError::InvalidRustCodegen);
                }
            }
            _ => return Err(PlanError::InvalidRustCodegen),
        }

        let hw = self
            .hardware
            .cpu
            .ok_or(PlanError::CpuMappingOnCpuAbsentHardware)?;
        if !hw.features.contains(cpu.required_features) {
            return Err(PlanError::CpuFeatureContractMismatch);
        }
        if hw.max_vector_bytes > 0 && cpu.simd_bytes > hw.max_vector_bytes {
            return Err(PlanError::CpuFeatureContractMismatch);
        }

        // Operand-specific access policies are fully resolved offline. Validate
        // their alignment/cache contracts once so native kernels can issue exact
        // loads/stores/prefetches without replay-time branching.
        let mut access_index = 0usize;
        while access_index < cpu.memory_access.len() {
            let access: CpuMemoryAccessPlan = cpu.memory_access[access_index];
            if access.required_alignment_bytes == 0
                || !access.required_alignment_bytes.is_power_of_two()
                || (access.prefetch_kind == PrefetchKind::None
                    && access.prefetch_distance_bytes != 0)
                || (access.prefetch_kind != PrefetchKind::None
                    && access.prefetch_distance_bytes == 0)
            {
                return Err(PlanError::InvalidCpuTuning);
            }
            match access.vector_kind {
                CpuVectorAccessKind::Scalar => {
                    if access.vector_bytes != 0 {
                        return Err(PlanError::InvalidCpuTuning);
                    }
                }
                CpuVectorAccessKind::AlignedVector => {
                    if access.vector_bytes == 0
                        || !access.vector_bytes.is_power_of_two()
                        || access.required_alignment_bytes < access.vector_bytes
                    {
                        return Err(PlanError::InvalidCpuTuning);
                    }
                }
                CpuVectorAccessKind::UnalignedVector => {
                    if access.vector_bytes == 0 || !access.vector_bytes.is_power_of_two() {
                        return Err(PlanError::InvalidCpuTuning);
                    }
                }
            }
            if access.vector_bytes > 0
                && hw.max_vector_bytes > 0
                && access.vector_bytes > hw.max_vector_bytes
            {
                return Err(PlanError::CpuFeatureContractMismatch);
            }
            let mut prior = 0usize;
            while prior < access_index {
                if cpu.memory_access[prior].storage == access.storage {
                    return Err(PlanError::InvalidCpuTuning);
                }
                prior += 1;
            }
            let storage = self.find_storage(access.storage).ok_or(PlanError::MissingStorage)?;
            let in_io = cpu.io.input_storages.contains(&access.storage)
                || cpu.io.output_storages.contains(&access.storage);
            if !in_io {
                return Err(PlanError::InvalidCpuTuning);
            }
            let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
            let alignment = access.required_alignment_bytes as usize;
            let storage_address = slot
                .offset_bytes
                .checked_add(storage.base_offset_bytes)
                .ok_or(PlanError::InvalidCpuTuning)?;
            if slot.alignment_bytes < alignment
                || storage_address % alignment != 0
                || (storage.physical_layout.alignment_bytes as usize) < alignment
            {
                return Err(PlanError::InvalidCpuTuning);
            }
            if storage.access == StorageAccess::ReadOnly
                && access.store_policy != StorePolicy::Cached
            {
                return Err(PlanError::InvalidCpuTuning);
            }
            if storage.access == StorageAccess::WriteOnly
                && matches!(
                    access.prefetch_kind,
                    PrefetchKind::ReadL1
                        | PrefetchKind::ReadL2
                        | PrefetchKind::ReadL3
                        | PrefetchKind::ReadShared
                )
            {
                return Err(PlanError::InvalidCpuTuning);
            }
            if access.prefetch_kind == PrefetchKind::ReadShared
                && !cpu.required_features.contains(cpu_feature::MOVRS)
            {
                return Err(PlanError::CpuFeatureContractMismatch);
            }
            access_index += 1;
        }

        if cpu.prefetch_kind == PrefetchKind::ReadShared
            && !cpu.required_features.contains(cpu_feature::MOVRS)
        {
            return Err(PlanError::CpuFeatureContractMismatch);
        }
        if (cpu.simd_bytes > 0 && !cpu.simd_bytes.is_multiple_of(16))
            || cpu.vector_unroll == 0
            || cpu.scalar_unroll == 0
            || cpu.software_pipeline_stages == 0
            || cpu.tile_m == 0
            || cpu.tile_n == 0
            || cpu.tile_k == 0
            || cpu.micro_m == 0
            || cpu.micro_n == 0
            || cpu.micro_k == 0
            || !cpu.tile_m.is_multiple_of(cpu.micro_m)
            || !cpu.tile_n.is_multiple_of(cpu.micro_n)
            || !cpu.tile_k.is_multiple_of(cpu.micro_k)
            || (cpu.prefetch_kind == PrefetchKind::None && cpu.prefetch_distance_bytes != 0)
            || (cpu.prefetch_kind != PrefetchKind::None && cpu.prefetch_distance_bytes == 0)
            || ((cpu.pack_a || cpu.pack_b) && cpu.pack_alignment_bytes == 0)
            || (!(cpu.pack_a || cpu.pack_b) && cpu.pack_alignment_bytes != 0)
            || (cpu.persistent_packed_weights && !cpu.pack_b)
            || ((cpu.pack_a || cpu.pack_b) && cpu.workspace_slots.is_empty())
        {
            return Err(PlanError::InvalidCpuTuning);
        }

        match cpu.family {
            KernelFamily::CpuDotSimd
            | KernelFamily::CpuVecMatSimd
            | KernelFamily::CpuVecMatAxpy
            | KernelFamily::CpuMatVecSimd
            | KernelFamily::CpuSgemmFixedTiny
            | KernelFamily::CpuSgemmBatchedSmall
            | KernelFamily::CpuSgemmDirect
            | KernelFamily::CpuSgemmDirectAxpy => {
                if cpu.pack_a || cpu.pack_b || cpu.persistent_packed_weights {
                    return Err(PlanError::InvalidCpuTuning);
                }
            }
            KernelFamily::CpuSgemmPacked4x8 => {
                if !cpu.pack_a || !cpu.pack_b || cpu.micro_m != 4 || cpu.micro_n != 8 {
                    return Err(PlanError::InvalidCpuTuning);
                }
            }
            KernelFamily::CpuSgemmPacked6xSimd => {
                if !cpu.pack_a
                    || !cpu.pack_b
                    || cpu.persistent_packed_weights
                    || cpu.micro_m != 6
                    || cpu.simd_bytes < 16
                    || !cpu.simd_bytes.is_multiple_of(4)
                    || cpu.micro_n != cpu.simd_bytes / 4
                {
                    return Err(PlanError::InvalidCpuTuning);
                }
            }
            KernelFamily::CpuSgemmPrepackedB4x8
                if (!cpu.pack_a
                    || !cpu.pack_b
                    || !cpu.persistent_packed_weights
                    || cpu.micro_m != 4
                    || cpu.micro_n != 8)
                => {
                    return Err(PlanError::InvalidCpuTuning);
                }
            _ => {}
        }

        // Scalable-vector machines still bind an exact vector length for an exact
        // Frozen Plan. Replay never chooses a different SVE/SME vector length.
        let uses_sve = cpu.required_features.contains(cpu_feature::SVE)
            || cpu.required_features.contains(cpu_feature::SVE2)
            || cpu.required_features.contains(cpu_feature::SVE_BF16)
            || cpu.required_features.contains(cpu_feature::SVE_I8MM)
            || cpu.required_features.contains(cpu_feature::SVE_F32MM)
            || cpu.required_features.contains(cpu_feature::SVE_F64MM);
        if uses_sve && (hw.sve_vector_bytes == 0 || cpu.simd_bytes != hw.sve_vector_bytes) {
            return Err(PlanError::CpuFeatureContractMismatch);
        }
        let uses_sme = cpu.required_features.contains(cpu_feature::SME)
            || cpu.required_features.contains(cpu_feature::SME2);
        if uses_sme && (hw.sme_vector_bytes == 0 || cpu.simd_bytes != hw.sme_vector_bytes) {
            return Err(PlanError::CpuFeatureContractMismatch);
        }

        let uses_rvv = cpu.required_features.contains(cpu_feature::RVV);
        if uses_rvv && (hw.rvv_vector_bytes == 0 || cpu.simd_bytes != hw.rvv_vector_bytes) {
            return Err(PlanError::CpuFeatureContractMismatch);
        }

        if cpu.pack_alignment_bytes > 0 && !cpu.pack_alignment_bytes.is_power_of_two() {
            return Err(PlanError::InvalidMemoryAlignment);
        }
        if cpu.pack_a || cpu.pack_b {
            let pack_alignment = usize::try_from(cpu.pack_alignment_bytes)
                .map_err(|_| PlanError::InvalidWorkspacePlan)?;
            let tile_m = usize::from(cpu.tile_m);
            let tile_n = usize::from(cpu.tile_n);
            let tile_k = usize::from(cpu.tile_k);
            let mut required = 0usize;
            if cpu.pack_a {
                required = required
                    .checked_add(
                        tile_m
                            .checked_mul(tile_k)
                            .and_then(|v| v.checked_mul(core::mem::size_of::<f32>()))
                            .ok_or(PlanError::InvalidWorkspacePlan)?,
                    )
                    .ok_or(PlanError::InvalidWorkspacePlan)?;
            }
            if cpu.pack_b {
                required = required
                    .checked_add(
                        tile_k
                            .checked_mul(tile_n)
                            .and_then(|v| v.checked_mul(core::mem::size_of::<f32>()))
                            .ok_or(PlanError::InvalidWorkspacePlan)?,
                    )
                    .ok_or(PlanError::InvalidWorkspacePlan)?;
            }
            let mut available = 0usize;
            let mut w = 0usize;
            while w < cpu.workspace_slots.len() {
                let slot = self.find_slot(cpu.workspace_slots[w]).ok_or(PlanError::MissingMemorySlot)?;
                if slot.alignment_bytes < pack_alignment || slot.offset_bytes % pack_alignment != 0 {
                    return Err(PlanError::InvalidWorkspacePlan);
                }
                available = available.checked_add(slot.bytes).ok_or(PlanError::InvalidWorkspacePlan)?;
                w += 1;
            }
            if available < required {
                return Err(PlanError::InvalidWorkspacePlan);
            }
        }
        if matches!(cpu.numa_policy, NumaPolicy::Fixed)
            && cpu.fixed_numa_node >= hw.numa_node_count
        {
            return Err(PlanError::InvalidCpuWorkerPlan);
        }
        match cpu.parallel_kind {
            CpuParallelKind::SingleThread => {
                if cpu.workers.len() != 1 || cpu.dispatch_mode != CpuDispatchMode::CallerThread {
                    return Err(PlanError::InvalidCpuWorkerPlan);
                }
            }
            CpuParallelKind::StaticShard
            | CpuParallelKind::Static2D
            | CpuParallelKind::Static3D => {
                if cpu.workers.is_empty()
                    || cpu.dispatch_mode != CpuDispatchMode::PersistentWorkerTeam
                {
                    return Err(PlanError::InvalidCpuWorkerPlan);
                }
            }
        }

        let mut i = 0usize;
        while i < cpu.workers.len() {
            let worker = &cpu.workers[i];
            if worker.worker_id != u16::try_from(i).map_err(|_| PlanError::InvalidCpuWorkerPlan)? {
                return Err(PlanError::InvalidCpuWorkerPlan);
            }
            if worker.work.begin > worker.work.end {
                return Err(PlanError::InvalidWorkRange);
            }
            let proc = self
                .find_cpu_processor(worker.processor_group, worker.logical_cpu_id)
                .ok_or(PlanError::InvalidCpuWorkerPlan)?;
            if worker.numa_node != proc.numa_node {
                return Err(PlanError::InvalidCpuWorkerPlan);
            }
            if matches!(cpu.numa_policy, NumaPolicy::Fixed)
                && worker.numa_node != cpu.fixed_numa_node
            {
                return Err(PlanError::InvalidCpuWorkerPlan);
            }
            let mut j = i + 1;
            while j < cpu.workers.len() {
                let other = &cpu.workers[j];
                if worker.worker_id == other.worker_id {
                    return Err(PlanError::DuplicateCpuWorker);
                }
                if worker.work.begin < other.work.end && other.work.begin < worker.work.end {
                    return Err(PlanError::InvalidCpuWorkerPlan);
                }
                j += 1;
            }
            i += 1;
        }

        Ok(())
    }

}

// =============================================================================
// Gpu Mapping validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_gpu_mapping(&self, node: &FrozenNode, gpu: &GpuKernelPlan) -> Result<(), PlanError> {
        // Geometric counts supplied by a signed but potentially malformed plan
        // MUST NOT overflow in debug or silently wrap in release mode.
        // These are one-time plan-validation checks, never hot-path scheduling.
        gpu.grid.checked_product().ok_or(PlanError::InvalidGpuGrid)?;
        let block_threads = gpu.block.checked_product().ok_or(PlanError::InvalidGpuBlock)?;
        let cluster_blocks = gpu.cluster.checked_product().ok_or(PlanError::InvalidGpuCluster)?;
        gpu.preferred_cluster.checked_product().ok_or(PlanError::InvalidClusterPolicy)?;
        self.validate_artifact_ref(gpu.artifact)?;
        self.validate_library_algorithm(gpu.artifact.kind, gpu.library_algorithm)?;
        if let Some(algorithm) = gpu.library_algorithm {
            if !matches!(
                algorithm.provider,
                LibraryProvider::Cublas
                    | LibraryProvider::CublasLt
                    | LibraryProvider::CuFft
                    | LibraryProvider::CuSolver
                    | LibraryProvider::Cudnn
                    | LibraryProvider::CuTensor
                    | LibraryProvider::CuSparse
                    | LibraryProvider::Nccl
                    | LibraryProvider::TargetNative
            ) || algorithm.provider_thread_count != 0
                || !algorithm.provider_thread_bindings.is_empty()
            {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            let provider_matches_operation = match algorithm.provider {
                LibraryProvider::Cublas | LibraryProvider::CublasLt => {
                    matches!(node.operation, Operation::MatMul { .. } | Operation::TensorContract { .. })
                }
                LibraryProvider::CuFft => matches!(node.operation, Operation::FourierTransform(_)),
                LibraryProvider::CuSolver => matches!(node.operation, Operation::DenseLinearAlgebra(_)),
                LibraryProvider::Cudnn => matches!(
                    node.operation,
                    Operation::Convolution(_)
                        | Operation::Attention(_)
                        | Operation::Normalization(_)
                        | Operation::Pooling(_)
                        | Operation::Softmax { .. }
                ),
                LibraryProvider::CuTensor => matches!(node.operation, Operation::TensorContract { .. }),
                LibraryProvider::CuSparse => matches!(
                    node.operation,
                    Operation::MatMul { .. } | Operation::TensorContract { .. } | Operation::Custom { .. }
                ),
                LibraryProvider::Nccl => matches!(node.operation, Operation::Collective(_)),
                LibraryProvider::TargetNative => true,
                _ => false,
            };
            if !provider_matches_operation {
                return Err(PlanError::InvalidLibraryAlgorithm);
            }
            if let Some(workspace) = algorithm.workspace_slot {
                if !gpu.workspace_slots.contains(&workspace) {
                    return Err(PlanError::InvalidWorkspacePlan);
                }
            }
        }
        if (gpu.artifact.kind == ArtifactKind::LibraryKernel) != (gpu.family == KernelFamily::VendorExact)
            || matches!(
                gpu.family,
                KernelFamily::CpuSgemmFixedTiny
                    | KernelFamily::CpuSgemmBatchedSmall
                    | KernelFamily::CpuSgemmDirect
                    | KernelFamily::CpuSgemmDirectAxpy
                    | KernelFamily::CpuSgemmPacked4x8
                    | KernelFamily::CpuSgemmPacked6xSimd
                    | KernelFamily::CpuSgemmPrepackedB4x8
                    | KernelFamily::CpuLinalgSmallStatic
                    | KernelFamily::CpuLinalgBatched
            )
        {
            return Err(PlanError::InvalidKernelFamily);
        }

        let hw = self
            .find_gpu(gpu.device_ordinal)
            .ok_or(PlanError::GpuMappingOnGpuAbsentHardware)?;

        match (gpu.artifact.kind, gpu.gpu_codegen) {
            (ArtifactKind::LibraryKernel, None) => {}
            (ArtifactKind::Cubin, Some(codegen)) => {
                if codegen.target_arch != hw.native_arch
                    || codegen.target_compute_major != hw.compute_major
                    || codegen.target_compute_minor != hw.compute_minor
                {
                    return Err(PlanError::GpuComputeCapabilityMismatch);
                }
                if codegen.toolchain_version.is_empty()
                    || codegen.target_arch.is_empty()
                    || codegen.compiled_registers_per_thread == 0
                    || codegen.compiled_max_threads_per_block == 0
                    || codegen.compiled_active_blocks_per_sm == 0
                    || codegen.compiled_active_blocks_per_sm > hw.max_blocks_per_sm
                    || codegen.compiled_registers_per_thread != gpu.target_registers_per_thread
                    || (codegen.compiled_max_threads_per_block as u64) < block_threads
                    || (gpu.target_blocks_per_sm > 0
                        && codegen.compiled_active_blocks_per_sm != gpu.target_blocks_per_sm)
                    || codegen.compiled_active_blocks_per_sm < gpu.min_blocks_per_sm
                    || (codegen.max_registers > 0
                        && codegen.compiled_registers_per_thread > codegen.max_registers)
                    || (codegen.use_fast_math
                        && (!codegen.flush_denormals_to_zero
                            || codegen.precise_division
                            || codegen.precise_sqrt
                            || !codegen.fuse_multiply_add))
                    || (codegen.device_lto && !codegen.relocatable_device_code)
                    || codegen.ptxas_opt_level != 3
                    || !codegen.allow_expensive_optimizations
                    || !codegen.fast_compile_disabled
                    || (codegen.compileiq_tuned
                        && (codegen.source != GpuKernelSource::CudaCppAot
                            || (codegen.advanced_controls_hash_hi == 0
                                && codegen.advanced_controls_hash_lo == 0)))
                    || (!codegen.compileiq_tuned
                        && (codegen.advanced_controls_hash_hi != 0
                            || codegen.advanced_controls_hash_lo != 0))
                    || codegen.device_debug
                    || codegen.device_stack_protector
                    || codegen.tensor_memory_access_checks
                {
                    return Err(PlanError::InvalidGpuCodegen);
                }
            }
            _ => return Err(PlanError::InvalidGpuCodegen),
        }

        if !hw.features.contains(gpu.required_features) {
            return Err(PlanError::GpuFeatureContractMismatch);
        }
        match gpu.math_mode {
            KernelMathMode::Tf32 => {
                if !gpu.required_features.contains(gpu_feature::TF32) {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            }
            KernelMathMode::Fp16Accumulate => {
                if !gpu.required_features.contains(gpu_feature::FP16) {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            }
            KernelMathMode::Bf16Accumulate
                if !gpu.required_features.contains(gpu_feature::BF16) => {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            _ => {}
        }
        if let Some(codegen) = gpu.gpu_codegen {
            let relaxed_codegen = codegen.use_fast_math
                || codegen.flush_denormals_to_zero
                || !codegen.precise_division
                || !codegen.precise_sqrt;
            if relaxed_codegen
                && !matches!(
                    gpu.math_mode,
                    KernelMathMode::FastMath | KernelMathMode::TargetVerified
                )
            {
                return Err(PlanError::InvalidAccuracyContract);
            }
        }

        let requires = |feature: FeatureMask| -> Result<(), PlanError> {
            if !gpu.required_features.contains(feature) || !hw.features.contains(feature) {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
            Ok(())
        };

        match gpu.async_copy {
            AsyncCopyKind::None | AsyncCopyKind::SoftwarePrefetch => {
                if !gpu.tensor_maps.is_empty() {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            AsyncCopyKind::CpAsync | AsyncCopyKind::CpAsyncBulk | AsyncCopyKind::Stas => {
                if !gpu.tensor_maps.is_empty() {
                    return Err(PlanError::InvalidGpuTuning);
                }
                match gpu.async_copy {
                    AsyncCopyKind::CpAsync => requires(gpu_feature::CP_ASYNC)?,
                    AsyncCopyKind::CpAsyncBulk => requires(gpu_feature::BULK_ASYNC_COPY)?,
                    AsyncCopyKind::Stas => requires(gpu_feature::STAS)?,
                    _ => {}
                }
            }
            AsyncCopyKind::Tma | AsyncCopyKind::TmaMulticast => {
                requires(gpu_feature::TMA)?;
                if gpu.async_copy == AsyncCopyKind::TmaMulticast {
                    requires(gpu_feature::TMA_MULTICAST)?;
                    requires(gpu_feature::THREAD_BLOCK_CLUSTERS)?;
                }
                if gpu.tensor_maps.is_empty() {
                    return Err(PlanError::InvalidGpuTuning);
                }
                let mut map_index = 0usize;
                while map_index < gpu.tensor_maps.len() {
                    let map = gpu.tensor_maps[map_index];
                    if map.global_dims.is_empty()
                        || map.global_dims.len() > 5
                        || map.global_strides_bytes.len() + 1 != map.global_dims.len()
                        || map.element_strides.len() != map.global_dims.len()
                        || (map.descriptor_hash_hi == 0 && map.descriptor_hash_lo == 0)
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    let rank = map.global_dims.len();
                    match map.kind {
                        TensorMapKind::Tiled => {
                            if map.box_dims.len() != rank
                                || !map.im2col_lower.is_empty()
                                || !map.im2col_upper.is_empty()
                                || map.im2col_wide_lower != 0
                                || map.im2col_wide_upper != 0
                                || map.channels_per_pixel != 0
                                || map.pixels_per_column != 0
                            {
                                return Err(PlanError::InvalidGpuTuning);
                            }
                        }
                        TensorMapKind::Im2Col => {
                            requires(gpu_feature::TENSOR_MAP_IM2COL)?;
                            if !(3..=5).contains(&rank)
                                || !map.box_dims.is_empty()
                                || map.im2col_lower.len() != rank - 2
                                || map.im2col_upper.len() != rank - 2
                                || map.im2col_wide_lower != 0
                                || map.im2col_wide_upper != 0
                                || map.channels_per_pixel == 0
                                || map.channels_per_pixel > 256
                                || map.pixels_per_column == 0
                                || map.pixels_per_column > 1024
                            {
                                return Err(PlanError::InvalidGpuTuning);
                            }
                            let limit = if rank == 3 { 32768 } else if rank == 4 { 128 } else { 16 };
                            let mut s = 0usize;
                            while s < rank - 2 {
                                let lo = map.im2col_lower[s];
                                let hi = map.im2col_upper[s];
                                if lo < -limit || lo >= limit || hi < -limit || hi >= limit || hi < lo {
                                    return Err(PlanError::InvalidGpuTuning);
                                }
                                s += 1;
                            }
                        }
                        TensorMapKind::Im2ColWideW | TensorMapKind::Im2ColWideW128 => {
                            requires(gpu_feature::TENSOR_MAP_IM2COL_WIDE)?;
                            if hw.compute_major < 10
                                || !(3..=5).contains(&rank)
                                || !map.box_dims.is_empty()
                                || !map.im2col_lower.is_empty()
                                || !map.im2col_upper.is_empty()
                                || map.im2col_wide_upper < map.im2col_wide_lower
                                || map.channels_per_pixel == 0
                                || map.channels_per_pixel > 256
                                || map.pixels_per_column == 0
                                || map.pixels_per_column > 1024
                            {
                                return Err(PlanError::InvalidGpuTuning);
                            }
                        }
                    }
                    if map.interleave != TensorMapInterleave::None && rank < 3 {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    let mut t = 0usize;
                    while t < map.global_dims.len() {
                        let invalid_box = map.kind == TensorMapKind::Tiled
                            && (map.box_dims[t] == 0 || map.box_dims[t] > 256);
                        if map.global_dims[t] == 0
                            || map.global_dims[t] > (u32::MAX as u64) + 1
                            || invalid_box
                            || map.element_strides[t] == 0
                            || map.element_strides[t] > 8
                        {
                            return Err(PlanError::InvalidGpuTuning);
                        }
                        t += 1;
                    }
                    t = 0;
                    while t < map.global_strides_bytes.len() {
                        let stride = map.global_strides_bytes[t];
                        let stride_alignment = if map.interleave == TensorMapInterleave::Bytes32 {
                            32
                        } else {
                            16
                        };
                        if stride == 0
                            || stride >= (1u64 << 40)
                            || !stride.is_multiple_of(stride_alignment)
                        {
                            return Err(PlanError::InvalidGpuTuning);
                        }
                        t += 1;
                    }
                    let storage = self.find_storage(map.storage).ok_or(PlanError::MissingStorage)?;
                    let arena = self.arena_of_storage(map.storage).ok_or(PlanError::MissingArena)?;
                    let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
                    let required_alignment = if map.interleave == TensorMapInterleave::Bytes32 {
                        32
                    } else {
                        16
                    };
                    let storage_address = slot
                        .offset_bytes
                        .checked_add(storage.base_offset_bytes)
                        .ok_or(PlanError::InvalidGpuTuning)?;
                    if !self.arena_accessible_on_gpu(arena, gpu.device_ordinal)
                        || slot.alignment_bytes < required_alignment
                        || storage_address % required_alignment != 0
                        || (!gpu.io.input_storages.contains(&map.storage)
                            && !gpu.io.output_storages.contains(&map.storage))
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    if map.oob_fill == TensorMapOobFill::NanRequestZeroFma {
                        let tensor = self
                            .find_tensor(storage.tensor)
                            .ok_or(PlanError::MissingTensor)?;
                        let is_float = matches!(tensor.dtype, DType::F32);
                        if !is_float {
                            return Err(PlanError::InvalidGpuTuning);
                        }
                    }
                    if map.oob_fill != TensorMapOobFill::None
                        && map.direction != TensorMapDirection::GlobalToShared
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    if map.store_reduction != TensorMapStoreReduction::None {
                        if map.direction != TensorMapDirection::SharedToGlobal
                            || !matches!(gpu.async_copy, AsyncCopyKind::Tma | AsyncCopyKind::TmaMulticast)
                        {
                            return Err(PlanError::InvalidGpuTuning);
                        }
                        let tensor = self
                            .find_tensor(storage.tensor)
                            .ok_or(PlanError::MissingTensor)?;
                        if matches!(
                            map.store_reduction,
                            TensorMapStoreReduction::BitAnd | TensorMapStoreReduction::BitOr
                        ) && matches!(tensor.dtype, DType::F32)
                        {
                            return Err(PlanError::InvalidGpuTuning);
                        }
                    }
                    let element_bytes = self
                        .find_tensor(storage.tensor)
                        .ok_or(PlanError::MissingTensor)?
                        .dtype
                        .size_bytes();
                    let inner_elements = match map.kind {
                        TensorMapKind::Tiled => map.box_dims[0] as usize,
                        TensorMapKind::Im2Col | TensorMapKind::Im2ColWideW | TensorMapKind::Im2ColWideW128 => {
                            map.channels_per_pixel as usize
                        }
                    };
                    let inner_bytes = inner_elements
                        .checked_mul(element_bytes)
                        .ok_or(PlanError::InvalidGpuTuning)?;
                    if map.kind == TensorMapKind::Tiled
                        && map.interleave == TensorMapInterleave::None
                        && inner_bytes % 16 != 0
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    if map.interleave == TensorMapInterleave::Bytes32
                        && map.swizzle != TensorMapSwizzle::Bytes32
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    let swizzle_bytes = match map.swizzle {
                        TensorMapSwizzle::None => 0usize,
                        TensorMapSwizzle::Bytes32 => 32,
                        TensorMapSwizzle::Bytes64 => 64,
                        TensorMapSwizzle::Bytes128
                        | TensorMapSwizzle::Bytes128Atom32
                        | TensorMapSwizzle::Bytes128Atom32Flip8
                        | TensorMapSwizzle::Bytes128Atom64 => 128,
                    };
                    if map.interleave == TensorMapInterleave::None
                        && swizzle_bytes > 0
                        && inner_bytes > swizzle_bytes
                    {
                        return Err(PlanError::InvalidGpuTuning);
                    }
                    map_index += 1;
                }
            }
        }
        match gpu.barrier {
            BarrierKind::None | BarrierKind::WarpSync | BarrierKind::BlockBarrier => {}
            BarrierKind::AsyncMBarrier => requires(gpu_feature::ASYNC_BARRIER)?,
            BarrierKind::ClusterBarrier => requires(gpu_feature::THREAD_BLOCK_CLUSTERS)?,
        }
        match gpu.matrix_acceleration {
            MatrixAcceleration::None | MatrixAcceleration::TargetNative => {}
            MatrixAcceleration::TensorCore
            | MatrixAcceleration::Wmma
            | MatrixAcceleration::Mma => requires(gpu_feature::TENSOR_CORES)?,
            MatrixAcceleration::Wgmma => requires(gpu_feature::WGMMA)?,
            MatrixAcceleration::Tcgen05 => {
                requires(gpu_feature::TCGEN05)?;
                requires(gpu_feature::TENSOR_MEMORY)?;
            }
            MatrixAcceleration::BlockScaledMma => {
                requires(gpu_feature::BLOCK_SCALED_MMA)?;
                // Current fifth-generation block-scaled MMA uses TMEM on supported
                // Blackwell-family targets. The exact low-precision format remains
                // part of tensor dtype/artifact metadata.
                if gpu.required_features.contains(gpu_feature::TCGEN05) {
                    requires(gpu_feature::TENSOR_MEMORY)?;
                }
            }
            MatrixAcceleration::SparseMma => {
                requires(gpu_feature::TENSOR_CORES)?;
                requires(gpu_feature::STRUCTURED_SPARSE_MMA)?;
            }
            MatrixAcceleration::SparseTcgen05 => {
                requires(gpu_feature::TCGEN05)?;
                requires(gpu_feature::TENSOR_MEMORY)?;
                requires(gpu_feature::STRUCTURED_SPARSE_MMA)?;
            }
            MatrixAcceleration::SparseBlockScaledMma => {
                requires(gpu_feature::TCGEN05)?;
                requires(gpu_feature::TENSOR_MEMORY)?;
                requires(gpu_feature::BLOCK_SCALED_MMA)?;
                requires(gpu_feature::STRUCTURED_SPARSE_MMA)?;
            }
        }
        let uses_tmem = gpu.tmem_columns_per_cta > 0
            || gpu.tmem_accumulator_stages > 0
            || gpu.tmem_operand_a;
        if uses_tmem {
            requires(gpu_feature::TENSOR_MEMORY)?;
            if gpu.tmem_columns_per_cta == 0
                || gpu.tmem_accumulator_stages == 0
                || gpu.tmem_columns_per_cta > hw.max_tmem_columns_per_cta
            {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
            if gpu.tmem_exclusive {
                if hw.tmem_exclusive_columns == 0
                    || gpu.tmem_columns_per_cta != hw.tmem_exclusive_columns
                {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            } else if !gpu.tmem_columns_per_cta.is_multiple_of(hw.tmem_allocation_granularity)
                || !gpu.tmem_columns_per_cta.is_power_of_two()
            {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
        } else if gpu.tmem_exclusive {
            return Err(PlanError::GpuFeatureContractMismatch);
        }
        if matches!(
            gpu.matrix_acceleration,
            MatrixAcceleration::Tcgen05
                | MatrixAcceleration::SparseTcgen05
                | MatrixAcceleration::SparseBlockScaledMma
        ) && !uses_tmem
        {
            return Err(PlanError::GpuFeatureContractMismatch);
        }
        if gpu.producer_warps > 0 || gpu.consumer_warps > 0 {
            requires(gpu_feature::WARP_SPECIALIZATION)?;
        }
        if gpu.cluster_launch {
            requires(gpu_feature::THREAD_BLOCK_CLUSTERS)?;
        }
        match gpu.graph_replay {
            GraphReplayMode::None => {}
            GraphReplayMode::HostGraph => requires(gpu_feature::CUDA_GRAPH)?,
            GraphReplayMode::DeviceTailLaunch
            | GraphReplayMode::DeviceFireAndForget
            | GraphReplayMode::DeviceSiblingLaunch => {
                requires(gpu_feature::CUDA_GRAPH)?;
                requires(gpu_feature::DEVICE_GRAPH_LAUNCH)?;
            }
        }
        if let Some(stream_id) = gpu.graph_preupload_stream {
            requires(gpu_feature::CUDA_GRAPH)?;
            if gpu.graph_replay == GraphReplayMode::None {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
            let stream = self.find_stream(stream_id).ok_or(PlanError::MissingStream)?;
            if stream.device_ordinal != gpu.device_ordinal
                || stream.execution_context != gpu.execution_context
            {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        }
        match gpu.shared_memory_mode {
            SharedMemoryMode::Standard => {
                if gpu.dynamic_shared_bytes > hw.max_dynamic_shared_bytes_per_block {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            SharedMemoryMode::RequirePortable => {
                requires(gpu_feature::SHARED_MEMORY_MODE)?;
                if gpu.dynamic_shared_bytes > hw.max_portable_dynamic_shared_bytes_per_block {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            SharedMemoryMode::AllowNonPortable => {
                requires(gpu_feature::SHARED_MEMORY_MODE)?;
                requires(gpu_feature::NON_PORTABLE_SHARED_MEMORY_MODE)?;
                if gpu.dynamic_shared_bytes > hw.max_dynamic_shared_bytes_per_block {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            SharedMemoryMode::AllowOversized | SharedMemoryMode::PreferOversized => {
                requires(gpu_feature::SHARED_MEMORY_MODE)?;
                requires(gpu_feature::OVERSIZED_SHARED_MEMORY)?;
                if hw.max_oversized_dynamic_shared_bytes_per_block == 0
                    || gpu.dynamic_shared_bytes > hw.max_oversized_dynamic_shared_bytes_per_block
                {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
        }

        // Per-storage cache/load/store policy is selected offline and becomes
        // part of the artifact identity. Validate it once so replay can use the
        // exact instruction stream with no policy arbitration.
        let mut access_index = 0usize;
        while access_index < gpu.memory_access.len() {
            let access = gpu.memory_access[access_index];
            if access.vector_bytes == 0
                || !access.vector_bytes.is_power_of_two()
                || access.vector_bytes > 32
                || access.required_alignment_bytes == 0
                || !access.required_alignment_bytes.is_power_of_two()
                || access.required_alignment_bytes < access.vector_bytes
                || !matches!(access.l2_prefetch_bytes, 0 | 64 | 128 | 256)
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            let mut prior = 0usize;
            while prior < access_index {
                if gpu.memory_access[prior].storage == access.storage {
                    return Err(PlanError::InvalidGpuTuning);
                }
                prior += 1;
            }
            let storage = self.find_storage(access.storage).ok_or(PlanError::MissingStorage)?;
            let in_io = gpu.io.input_storages.contains(&access.storage)
                || gpu.io.output_storages.contains(&access.storage);
            if !in_io {
                return Err(PlanError::InvalidGpuTuning);
            }
            let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
            let alignment = access.required_alignment_bytes as usize;
            let storage_address = slot
                .offset_bytes
                .checked_add(storage.base_offset_bytes)
                .ok_or(PlanError::InvalidGpuTuning)?;
            if slot.alignment_bytes < alignment
                || storage_address % alignment != 0
                || (storage.physical_layout.alignment_bytes as usize) < alignment
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            if storage.access == StorageAccess::ReadOnly
                && access.store_cache != GpuStoreCachePolicy::Default
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            if storage.access == StorageAccess::WriteOnly
                && (access.load_cache != GpuLoadCachePolicy::Default || access.l2_prefetch_bytes != 0)
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            if access.l2_prefetch_bytes != 0 {
                let Some(codegen) = gpu.gpu_codegen else {
                    return Err(PlanError::InvalidGpuTuning);
                };
                if (codegen.ptx_isa_major, codegen.ptx_isa_minor) < (7, 0)
                    || hw.compute_major < 8
                {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            }
            if access.load_cache == GpuLoadCachePolicy::UniformReadOnly
                && storage.access != StorageAccess::ReadOnly
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            if access.load_cache == GpuLoadCachePolicy::NonCoherent && hw.compute_major < 4 {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
            let eviction_requested = access.l1_eviction != GpuCacheEvictionPriority::Default
                || access.l2_eviction != GpuCacheEvictionPriority::Default;
            if eviction_requested && hw.compute_major < 7 {
                return Err(PlanError::GpuFeatureContractMismatch);
            }
            if access.l2_eviction == GpuCacheEvictionPriority::NoAllocate {
                return Err(PlanError::InvalidGpuTuning);
            }
            if access.l2_eviction != GpuCacheEvictionPriority::Default {
                let Some(codegen) = gpu.gpu_codegen else {
                    return Err(PlanError::InvalidGpuTuning);
                };
                if (codegen.ptx_isa_major, codegen.ptx_isa_minor) < (7, 4)
                    || hw.compute_major < 8
                {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }
            }
            access_index += 1;
        }

        // Constant memory is a one-time bound residency choice for small, read-only
        // broadcast coefficients/parameters. It is never populated during replay.
        let mut const_index = 0usize;
        let mut const_end_max = 0usize;
        while const_index < gpu.constant_memory.len() {
            let entry = gpu.constant_memory[const_index];
            if entry.bytes == 0 {
                return Err(PlanError::InvalidGpuTuning);
            }
            let storage = self.find_storage(entry.storage).ok_or(PlanError::MissingStorage)?;
            if storage.access != StorageAccess::ReadOnly
                || usize::try_from(entry.bytes).map_err(|_| PlanError::InvalidGpuTuning)? > storage.physical_bytes
                || (!gpu.io.input_storages.contains(&entry.storage))
            {
                return Err(PlanError::InvalidGpuTuning);
            }
            let begin = entry.symbol_offset_bytes as usize;
            let end = begin
                .checked_add(entry.bytes as usize)
                .ok_or(PlanError::InvalidGpuTuning)?;
            if end > hw.constant_memory_bytes {
                return Err(PlanError::InvalidGpuTuning);
            }
            let mut prior = 0usize;
            while prior < const_index {
                let other = gpu.constant_memory[prior];
                if other.storage == entry.storage {
                    return Err(PlanError::InvalidGpuTuning);
                }
                let other_begin = other.symbol_offset_bytes as usize;
                let other_end = other_begin
                    .checked_add(other.bytes as usize)
                    .ok_or(PlanError::InvalidGpuTuning)?;
                if begin < other_end && other_begin < end {
                    return Err(PlanError::InvalidGpuTuning);
                }
                prior += 1;
            }
            const_end_max = const_end_max.max(end);
            const_index += 1;
        }
        if const_end_max > 0 && hw.constant_memory_bytes == 0 {
            return Err(PlanError::GpuFeatureContractMismatch);
        }

        if let Some(policy) = gpu.l2_access_policy {
            requires(gpu_feature::PERSISTING_L2)?;
            if policy.bytes == 0
                || policy.hit_ratio_per_mille > 1000
                || (hw.max_access_policy_window_bytes > 0
                    && policy.bytes > hw.max_access_policy_window_bytes)
                || ((policy.hit == L2AccessProperty::Persisting
                    || policy.miss == L2AccessProperty::Persisting)
                    && hw.max_persisting_l2_bytes == 0)
            {
                return Err(PlanError::InvalidL2AccessPolicy);
            }
            if policy.hit == L2AccessProperty::Persisting
                || policy.miss == L2AccessProperty::Persisting
            {
                let context = self
                    .find_context(gpu.execution_context)
                    .ok_or(PlanError::InvalidGpuExecutionContext)?;
                if context.persisting_l2_reservation_bytes == 0 {
                    return Err(PlanError::InvalidL2AccessPolicy);
                }
            }
            let storage = self
                .find_storage(policy.storage)
                .ok_or(PlanError::MissingStorage)?;
            let arena = self
                .arena_of_storage(policy.storage)
                .ok_or(PlanError::MissingArena)?;
            let end = policy
                .offset_bytes
                .checked_add(policy.bytes)
                .ok_or(PlanError::InvalidL2AccessPolicy)?;
            let arena_on_device =
                if arena.device_allocation == DeviceAllocationPolicy::MulticastMapped {
                arena.multicast_devices.contains(&gpu.device_ordinal)
            } else {
                arena.device_ordinal == Some(gpu.device_ordinal)
            };
            if !arena_on_device || end > storage.physical_bytes {
                return Err(PlanError::InvalidL2AccessPolicy);
            }
        }
        if let Some(domain) = gpu.memory_sync_domain {
            requires(gpu_feature::MEMORY_SYNC_DOMAINS)?;
            if domain.logical_domain > 1 || domain.physical_domain >= hw.memory_sync_domain_count {
                return Err(PlanError::InvalidMemorySyncDomain);
            }
        }
        if gpu.cooperative_launch {
            requires(gpu_feature::COOPERATIVE_LAUNCH)?;
        }
        if gpu.programmatic_dependent_launch || gpu.programmatic_stream_serialization
            || gpu.programmatic_event.is_some()
        {
            requires(gpu_feature::PROGRAMMATIC_DEPENDENT_LAUNCH)?;
        }
        if gpu.programmatic_event_trigger_at_block_start && gpu.programmatic_event.is_none() {
            return Err(PlanError::InvalidGpuLaunchAttribute);
        }
        if let Some(event_id) = gpu.programmatic_event {
            let event = self.find_event(event_id).ok_or(PlanError::MissingEvent)?;
            if event.device_ordinal != gpu.device_ordinal
                || !event.disable_timing
                || event.interprocess
            {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        }
        if let Some(event_id) = gpu.launch_completion_event {
            requires(gpu_feature::LAUNCH_COMPLETION_EVENT)?;
            let event = self.find_event(event_id).ok_or(PlanError::MissingEvent)?;
            if event.device_ordinal != gpu.device_ordinal {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        }
        if let Some(priority) = gpu.launch_priority {
            requires(gpu_feature::KERNEL_LAUNCH_PRIORITY)?;
            if priority < hw.greatest_stream_priority
                || priority > hw.least_stream_priority
                || (gpu.graph_replay != GraphReplayMode::None && !gpu.graph_use_node_priority)
            {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        }
        if gpu.graph_use_node_priority {
            requires(gpu_feature::CUDA_GRAPH)?;
            if gpu.graph_replay == GraphReplayMode::None || gpu.launch_priority.is_none() {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        }
        if gpu.device_updatable_graph_node {
            requires(gpu_feature::CUDA_GRAPH)?;
            requires(gpu_feature::DEVICE_UPDATABLE_GRAPH_NODE)?;
            // Device-updatable nodes are graph-only, require single instantiation,
            // and are pre-uploaded so a device update never triggers first-use setup.
            if gpu.graph_replay == GraphReplayMode::None
                || gpu.graph_preupload_stream.is_none()
                || !gpu.graph_single_instantiation
            {
                return Err(PlanError::InvalidGpuLaunchAttribute);
            }
        } else if gpu.graph_single_instantiation && gpu.graph_replay == GraphReplayMode::None {
            return Err(PlanError::InvalidGpuLaunchAttribute);
        }
        if gpu.nvlink_util_centric_scheduling {
            requires(gpu_feature::NVLINK_UTIL_CENTRIC_SCHEDULING)?;
        }
        if gpu.portable_cluster_mode == PortableClusterMode::AllowNonPortable {
            requires(gpu_feature::NON_PORTABLE_CLUSTER_SIZE)?;
        }
        if !gpu.preferred_cluster.is_all_zero() {
            if !gpu.preferred_cluster.is_all_nonzero() {
                return Err(PlanError::InvalidClusterPolicy);
            }
            requires(gpu_feature::PREFERRED_CLUSTER_DIMENSION)?;
        }
        if gpu.cluster_scheduling != ClusterSchedulingPolicy::NotApplicable {
            requires(gpu_feature::THREAD_BLOCK_CLUSTERS)?;
        }
        if gpu.cluster_launch_control == ClusterLaunchControlMode::WorkStealing {
            requires(gpu_feature::CLUSTER_LAUNCH_CONTROL)?;
            if !gpu.cluster_launch || gpu.cluster.is_all_zero() {
                return Err(PlanError::InvalidGpuCluster);
            }
        }
        if gpu.use_distributed_shared_memory {
            requires(gpu_feature::DISTRIBUTED_SHARED_MEMORY)?;
            if !gpu.cluster_launch || gpu.cluster.is_all_zero() {
                return Err(PlanError::InvalidGpuCluster);
            }
        }

        if gpu.sm_partition.kind == SmPartitionKind::GreenContext {
            requires(gpu_feature::GREEN_CONTEXT)?;
        }
        if gpu.sm_partition.kind == SmPartitionKind::ExecutionAffinity {
            requires(gpu_feature::EXECUTION_AFFINITY)?;
        }
        if gpu.sm_partition.locality_domain.is_some() {
            requires(gpu_feature::LOCALITY_DOMAINS)?;
        }

        if gpu.vector_bytes == 0
            || !gpu.vector_bytes.is_power_of_two()
            || (gpu.async_copy != AsyncCopyKind::None && gpu.pipeline_stages == 0)
        {
            return Err(PlanError::InvalidGpuTuning);
        }
        if gpu.matrix_acceleration != MatrixAcceleration::None
            && (gpu.tile_m == 0
                || gpu.tile_n == 0
                || gpu.tile_k == 0
                || gpu.warp_m == 0
                || gpu.warp_n == 0
                || gpu.warp_k == 0
                || !gpu.tile_m.is_multiple_of(gpu.warp_m)
                || !gpu.tile_n.is_multiple_of(gpu.warp_n)
                || !gpu.tile_k.is_multiple_of(gpu.warp_k))
        {
            return Err(PlanError::InvalidGpuTuning);
        }
        match gpu.partition {
            GpuPartitionKind::Grid | GpuPartitionKind::Grouped | GpuPartitionKind::Ragged => {
                if gpu.split_k > 1 || gpu.stream_k_slices > 1 || gpu.persistent_ctas_per_sm != 0 {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            GpuPartitionKind::SplitK => {
                if gpu.split_k < 2 || gpu.stream_k_slices > 1 || gpu.persistent_ctas_per_sm != 0 {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            GpuPartitionKind::StreamK => {
                if gpu.stream_k_slices < 2 || gpu.split_k > 1 || gpu.persistent_ctas_per_sm != 0 {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
            GpuPartitionKind::PersistentCta => {
                if gpu.persistent_ctas_per_sm == 0
                    || gpu.persistent_ctas_per_sm > hw.max_blocks_per_sm
                {
                    return Err(PlanError::InvalidGpuTuning);
                }
            }
        }

        if gpu.grid.x == 0
            || gpu.grid.y == 0
            || gpu.grid.z == 0
            || gpu.grid.x > hw.max_grid_dim[0]
            || gpu.grid.y > hw.max_grid_dim[1]
            || gpu.grid.z > hw.max_grid_dim[2]
        {
            return Err(PlanError::InvalidGpuGrid);
        }
        if gpu.block.x == 0
            || gpu.block.y == 0
            || gpu.block.z == 0
            || gpu.block.x > hw.max_block_dim[0]
            || gpu.block.y > hw.max_block_dim[1]
            || gpu.block.z > hw.max_block_dim[2]
        {
            return Err(PlanError::InvalidGpuBlock);
        }
        if gpu.max_threads_per_block == 0
            || gpu.max_threads_per_block > hw.max_threads_per_block
            || block_threads > gpu.max_threads_per_block as u64
            || block_threads > hw.max_threads_per_block as u64
            || gpu.target_registers_per_thread > hw.max_registers_per_thread
            || gpu.target_blocks_per_sm > hw.max_blocks_per_sm
            || gpu.min_blocks_per_sm > hw.max_blocks_per_sm
        {
            return Err(PlanError::InvalidGpuBlock);
        }
        // Hardware validation also requires warp_size != 0, but keep the
        // arithmetic locally safe against malformed catalog/hardware records.
        if hw.warp_size == 0 { return Err(PlanError::InvalidGpuWarpPlan); }
        let expected_warps = block_threads
            .checked_add(hw.warp_size as u64 - 1)
            .ok_or(PlanError::InvalidGpuWarpPlan)? / hw.warp_size as u64;
        if expected_warps != gpu.warps_per_block as u64
            || gpu.warps_per_block as u16 > hw.max_warps_per_sm
            || gpu.producer_warps as u16 + gpu.consumer_warps as u16
                > gpu.warps_per_block as u16
            || gpu.shared_memory_carveout_percent > 100
        {
            return Err(PlanError::InvalidGpuWarpPlan);
        }

        // The offline occupancy target is part of the frozen performance plan.
        // Reject impossible targets here so replay never performs occupancy search.
        let occupancy_blocks = if gpu.target_blocks_per_sm > 0 {
            gpu.target_blocks_per_sm
        } else {
            gpu.min_blocks_per_sm
        } as u64;
        if occupancy_blocks > 0 {
            let compiled_registers = gpu
                .gpu_codegen
                .map(|codegen| codegen.compiled_registers_per_thread)
                .unwrap_or(gpu.target_registers_per_thread);
            let compiled_static_shared = gpu
                .gpu_codegen
                .map(|codegen| codegen.compiled_static_shared_bytes as u64)
                .unwrap_or(0);
            let registers_per_block = block_threads
                .checked_mul(compiled_registers as u64)
                .ok_or(PlanError::InvalidGpuBlock)?;
            if registers_per_block > hw.max_registers_per_block as u64 {
                return Err(PlanError::InvalidGpuBlock);
            }
            let shared_per_block = compiled_static_shared
                .checked_add(gpu.dynamic_shared_bytes as u64)
                .ok_or(PlanError::InvalidGpuBlock)?;
            if shared_per_block > hw.max_shared_bytes_per_block as u64 {
                return Err(PlanError::InvalidGpuBlock);
            }
            if block_threads
                .checked_mul(occupancy_blocks)
                .ok_or(PlanError::InvalidGpuBlock)?
                > hw.max_threads_per_sm as u64
                || expected_warps
                    .checked_mul(occupancy_blocks)
                    .ok_or(PlanError::InvalidGpuWarpPlan)?
                    > hw.max_warps_per_sm as u64
                || registers_per_block
                    .checked_mul(occupancy_blocks)
                    .ok_or(PlanError::InvalidGpuBlock)?
                    > hw.registers_per_sm as u64
                || shared_per_block
                    .checked_mul(occupancy_blocks)
                    .ok_or(PlanError::InvalidGpuBlock)?
                    > hw.max_shared_bytes_per_sm as u64
            {
                return Err(PlanError::InvalidGpuBlock);
            }
        }
        if let Some(codegen) = gpu.gpu_codegen {
            let total_shared = (codegen.compiled_static_shared_bytes as u64)
                .checked_add(gpu.dynamic_shared_bytes as u64)
                .ok_or(PlanError::InvalidGpuTuning)?;
            if total_shared > hw.max_shared_bytes_per_block as u64
                || total_shared > hw.max_shared_bytes_per_sm as u64
            {
                return Err(PlanError::InvalidGpuTuning);
            }
        }

        if gpu.cluster_launch {
            if gpu.cluster.x == 0 || gpu.cluster.y == 0 || gpu.cluster.z == 0
                || gpu.portable_cluster_mode == PortableClusterMode::NotApplicable
                || gpu.cluster_scheduling == ClusterSchedulingPolicy::NotApplicable
            {
                return Err(PlanError::InvalidGpuCluster);
            }
            if !gpu.grid.x.is_multiple_of(gpu.cluster.x)
                || !gpu.grid.y.is_multiple_of(gpu.cluster.y)
                || !gpu.grid.z.is_multiple_of(gpu.cluster.z)
            {
                return Err(PlanError::InvalidGpuCluster);
            }
            if gpu.portable_cluster_mode == PortableClusterMode::RequirePortable
                && hw.max_portable_cluster_blocks > 0
                && cluster_blocks > hw.max_portable_cluster_blocks as u64
            {
                return Err(PlanError::InvalidClusterPolicy);
            }
            if !gpu.preferred_cluster.is_all_zero()
                && (!gpu.preferred_cluster.x.is_multiple_of(gpu.cluster.x)
                    || !gpu.preferred_cluster.y.is_multiple_of(gpu.cluster.y)
                    || !gpu.preferred_cluster.z.is_multiple_of(gpu.cluster.z))
                {
                    return Err(PlanError::InvalidClusterPolicy);
                }
        } else if !gpu.cluster.is_all_zero()
            || !gpu.preferred_cluster.is_all_zero()
            || gpu.cluster_scheduling != ClusterSchedulingPolicy::NotApplicable
            || gpu.portable_cluster_mode != PortableClusterMode::NotApplicable
        {
            return Err(PlanError::InvalidClusterPolicy);
        }

        let context = self
            .find_context(gpu.execution_context)
            .ok_or(PlanError::InvalidGpuExecutionContext)?;
        if context.device_ordinal != gpu.device_ordinal
            || context.sm_partition != gpu.sm_partition
        {
            return Err(PlanError::InvalidGpuExecutionContext);
        }

        let stream = self
            .find_stream(gpu.stream)
            .ok_or(PlanError::InvalidGpuStream)?;
        if stream.device_ordinal != gpu.device_ordinal
            || stream.execution_context != gpu.execution_context
        {
            return Err(PlanError::InvalidGpuStream);
        }

        if let Some(domain) = gpu.sm_partition.locality_domain {
            if domain >= hw.locality_domain_count {
                return Err(PlanError::InvalidGpuExecutionContext);
            }
        }
        if gpu.sm_partition.sm_count > hw.sm_count {
            return Err(PlanError::InvalidSmPartition);
        }

        Ok(())
    }

}

}

mod execution_checks {
    use super::*;

/// GPU-backed external arenas must bind to a device admitted by the frozen
/// memory topology, not merely to an arbitrary ordinal supplied by a caller.
/// Host-only external arenas deliberately have no GPU device requirement.
#[inline(always)]
fn external_arena_device_is_declared(
    device_ordinal: Option<u16>,
    frozen_memory_gpu_ordinals: &[u16],
) -> bool {
    device_ordinal.is_none_or(|device| frozen_memory_gpu_ordinals.contains(&device))
}

// =============================================================================
// Memory validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_memory(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.arenas.len() {
            let arena = &self.arenas[i];
            if arena.alignment_bytes == 0 || !arena.alignment_bytes.is_power_of_two() {
                return Err(PlanError::InvalidArenaAlignment);
            }
            match arena.page_policy {
                PagePolicy::BasePages => {
                    if arena.page_size_bytes != 0 {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
                PagePolicy::TransparentHugePages | PagePolicy::ExplicitHugePages => {
                    if arena.page_size_bytes == 0 || !arena.page_size_bytes.is_power_of_two() {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                    if arena.page_policy == PagePolicy::ExplicitHugePages
                        && (arena.alignment_bytes < arena.page_size_bytes
                            || !arena.bytes.is_multiple_of(arena.page_size_bytes))
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
            }
            if arena.numa_node.is_some() && !arena.numa_interleave_nodes.is_empty() {
                return Err(PlanError::InvalidArenaPolicy);
            }
            if let Some(node) = arena.numa_node {
                let cpu = self.hardware.cpu.ok_or(PlanError::InvalidArenaPolicy)?;
                if node >= cpu.numa_node_count
                    || (!self.canonical.memory_topology.host_numa_nodes.is_empty()
                        && !self.canonical.memory_topology.host_numa_nodes.contains(&node))
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
            }
            if arena.numa_interleave_nodes.len() == 1 {
                return Err(PlanError::InvalidArenaPolicy);
            }
            let mut numa_i = 0usize;
            while numa_i < arena.numa_interleave_nodes.len() {
                let cpu = self.hardware.cpu.ok_or(PlanError::InvalidArenaPolicy)?;
                let node = arena.numa_interleave_nodes[numa_i];
                if node >= cpu.numa_node_count
                    || (!self.canonical.memory_topology.host_numa_nodes.is_empty()
                        && !self.canonical.memory_topology.host_numa_nodes.contains(&node))
                {
                    return Err(PlanError::InvalidMemoryTopology);
                }
                if numa_i > 0 && arena.numa_interleave_nodes[numa_i - 1] >= node {
                    return Err(PlanError::InvalidArenaPolicy);
                }
                numa_i += 1;
            }
            let host_page_policy = !matches!(arena.page_policy, PagePolicy::BasePages)
                || arena.prefault
                || arena.numa_node.is_some()
                || !arena.numa_interleave_nodes.is_empty();
            if host_page_policy
                && !matches!(
                    arena.class,
                    MemoryClass::HostAligned
                        | MemoryClass::HostPinned
                        | MemoryClass::HostMappedZeroCopy
                        | MemoryClass::HostPageableDirect
                )
            {
                return Err(PlanError::InvalidArenaPolicy);
            }
            if arena.class != MemoryClass::UnifiedManaged && arena.unified_advice.is_some() {
                return Err(PlanError::InvalidArenaPolicy);
            }
            if arena.device_allocation == DeviceAllocationPolicy::StreamOrderedPool {
                if arena.memory_pool.is_none() {
                    return Err(PlanError::InvalidMemoryPool);
                }
            } else if arena.memory_pool.is_some() {
                return Err(PlanError::InvalidMemoryPool);
            }
            if arena.device_allocation == DeviceAllocationPolicy::MulticastMapped {
                if arena.multicast_devices.len() < 2 {
                    return Err(PlanError::InvalidArenaPolicy);
                }
            } else if !arena.multicast_devices.is_empty() {
                return Err(PlanError::InvalidArenaPolicy);
            }

            match arena.class {
                MemoryClass::HostAligned => {
                    if arena.host_policy != HostMemoryPolicy::Pageable
                        || arena.device_allocation != DeviceAllocationPolicy::NotApplicable
                        || arena.device_ordinal.is_some()
                        || arena.locality_domain.is_some()
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
                MemoryClass::HostPageableDirect => {
                    let device = arena.device_ordinal.ok_or(PlanError::InvalidArenaPolicy)?;
                    if arena.host_policy != HostMemoryPolicy::Pageable
                        || arena.device_allocation != DeviceAllocationPolicy::NotApplicable
                        || arena.locality_domain.is_some()
                        || !self.canonical.memory_topology.gpu_device_ordinals.contains(&device)
                        || !self.gpu_has_feature(device, gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS)
                        || !self.gpu_has_feature(
                            device,
                            gpu_feature::PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES,
                        )
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
                MemoryClass::HostPinned => {
                    if !self.canonical.memory_topology.supports_pinned_host_memory {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                    let valid = matches!(
                        arena.host_policy,
                        HostMemoryPolicy::Pinned
                            | HostMemoryPolicy::PinnedPortable
                            | HostMemoryPolicy::PinnedWriteCombined
                    );
                    if !valid
                        || arena.device_allocation != DeviceAllocationPolicy::NotApplicable
                        || arena.device_ordinal.is_some()
                        || arena.locality_domain.is_some()
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
                MemoryClass::HostMappedZeroCopy => {
                    if !self.canonical.memory_topology.supports_mapped_host_memory {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                    let valid = matches!(
                        arena.host_policy,
                        HostMemoryPolicy::MappedZeroCopy | HostMemoryPolicy::MappedWriteCombined
                    );
                    if !valid
                        || arena.device_allocation != DeviceAllocationPolicy::NotApplicable
                        || arena.device_ordinal.is_some()
                        || arena.locality_domain.is_some()
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                }
                MemoryClass::DeviceLocal => {
                    let device = arena.device_ordinal.ok_or(PlanError::InvalidArenaPolicy)?;
                    if !self.canonical.memory_topology.gpu_device_ordinals.contains(&device) {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                    let gpu = self.find_gpu(device).ok_or(PlanError::GpuDeviceMismatch)?;
                    if let Some(domain) = arena.locality_domain {
                        if domain >= gpu.locality_domain_count
                            || !gpu.features.contains(gpu_feature::LOCALITY_DOMAINS)
                        {
                            return Err(PlanError::InvalidArenaPolicy);
                        }
                    }
                    match arena.device_allocation {
                        DeviceAllocationPolicy::StreamOrderedPool => {
                            if !gpu.features.contains(gpu_feature::ASYNC_MEMORY_POOL) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        DeviceAllocationPolicy::GraphOwned => {
                            if !gpu.features.contains(gpu_feature::CUDA_GRAPH) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        DeviceAllocationPolicy::VirtualMemoryMapped => {
                            if !gpu.features.contains(gpu_feature::VIRTUAL_MEMORY_MANAGEMENT) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        DeviceAllocationPolicy::MulticastMapped => {
                            if arena.locality_domain.is_some()
                                || !gpu.features.contains(gpu_feature::VIRTUAL_MEMORY_MANAGEMENT)
                                || !gpu.features.contains(gpu_feature::MULTICAST_MEMORY)
                                || !arena.multicast_devices.contains(&device)
                                || arena.bytes == 0
                                || !arena.bytes.is_multiple_of(gpu.multicast_min_granularity_bytes)
                                || !arena.bytes.is_multiple_of(gpu.multicast_recommended_granularity_bytes)
                                || arena.alignment_bytes
                                    < gpu.multicast_recommended_granularity_bytes
                            {
                                return Err(PlanError::InvalidArenaPolicy);
                            }
                            let mut m = 0usize;
                            while m < arena.multicast_devices.len() {
                                let member = arena.multicast_devices[m];
                                if !self.canonical.memory_topology.gpu_device_ordinals.contains(&member) {
                                    return Err(PlanError::InvalidMemoryTopology);
                                }
                                let member_gpu = self
                                    .find_gpu(member)
                                    .ok_or(PlanError::GpuDeviceMismatch)?;
                                if !member_gpu
                                    .features
                                    .contains(gpu_feature::MULTICAST_MEMORY)
                                    || !member_gpu
                                        .features
                                        .contains(gpu_feature::VIRTUAL_MEMORY_MANAGEMENT)
                                    || member_gpu.multicast_min_granularity_bytes == 0
                                    || member_gpu.multicast_recommended_granularity_bytes == 0
                                    || !arena.bytes.is_multiple_of(member_gpu.multicast_min_granularity_bytes)
                                    || !arena.bytes.is_multiple_of(member_gpu.multicast_recommended_granularity_bytes)
                                {
                                    return Err(PlanError::GpuFeatureContractMismatch);
                                }
                                if m > 0 && arena.multicast_devices[m - 1] >= member {
                                    return Err(PlanError::InvalidArenaPolicy);
                                }
                                m += 1;
                            }
                        }
                        DeviceAllocationPolicy::FabricMapped => {
                            if !gpu.features.contains(gpu_feature::VIRTUAL_MEMORY_MANAGEMENT)
                                || !gpu.features.contains(gpu_feature::FABRIC_MEMORY)
                            {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        DeviceAllocationPolicy::ExtendedGpuMemory => {
                            if !gpu.features.contains(gpu_feature::EXTENDED_GPU_MEMORY) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        DeviceAllocationPolicy::Preallocated
                        | DeviceAllocationPolicy::External => {}
                        DeviceAllocationPolicy::NotApplicable => {
                            return Err(PlanError::InvalidArenaPolicy);
                        }
                    }
                }
                MemoryClass::UnifiedManaged => {
                    if !self.canonical.memory_topology.supports_unified_memory {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                    if arena.host_policy != HostMemoryPolicy::Pageable
                        || arena.locality_domain.is_some()
                        || matches!(
                            arena.device_allocation,
                            DeviceAllocationPolicy::NotApplicable
                                | DeviceAllocationPolicy::StreamOrderedPool
                                | DeviceAllocationPolicy::GraphOwned
                                | DeviceAllocationPolicy::VirtualMemoryMapped
                                | DeviceAllocationPolicy::MulticastMapped
                                | DeviceAllocationPolicy::FabricMapped
                                | DeviceAllocationPolicy::ExtendedGpuMemory
                        )
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                    if let Some(device) = arena.device_ordinal {
                        if !self.canonical.memory_topology.gpu_device_ordinals.contains(&device) {
                            return Err(PlanError::InvalidMemoryTopology);
                        }
                        if !self.gpu_has_feature(device, gpu_feature::UNIFIED_MEMORY) {
                            return Err(PlanError::GpuFeatureContractMismatch);
                        }
                    }
                    if let Some(advice) = arena.unified_advice {
                        if let Some(location) = advice.preferred_location {
                            self.validate_memory_location(location)?;
                        }
                        let mut a = 0usize;
                        while a < advice.accessed_by.len() {
                            self.validate_memory_location(advice.accessed_by[a])?;
                            if a > 0
                                && Self::memory_location_order_key(advice.accessed_by[a - 1])
                                    >= Self::memory_location_order_key(advice.accessed_by[a])
                            {
                                return Err(PlanError::InvalidArenaPolicy);
                            }
                            a += 1;
                        }
                    }
                }
                MemoryClass::External => {
                    if arena.device_allocation != DeviceAllocationPolicy::External
                        || arena.unified_advice.is_some()
                    {
                        return Err(PlanError::InvalidArenaPolicy);
                    }
                    // Unlike host-backed external arenas, GPU-backed arenas
                    // have an exact frozen device affinity. Reject ordinals
                    // outside the workload's admitted memory topology before
                    // a foreign address is ever bound or replayed. The
                    // topology itself is validated against real hardware by
                    // CanonicalWorkload::validate_topologies.
                    if !external_arena_device_is_declared(
                        arena.device_ordinal,
                        self.canonical.memory_topology.gpu_device_ordinals,
                    ) {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                }
            }
            i += 1;
        }

        i = 0;
        while i < self.memory_slots.len() {
            let slot = &self.memory_slots[i];
            if slot.alignment_bytes == 0 || !slot.alignment_bytes.is_power_of_two() {
                return Err(PlanError::InvalidMemoryAlignment);
            }
            if !slot.offset_bytes.is_multiple_of(slot.alignment_bytes) {
                return Err(PlanError::MemorySlotMisaligned);
            }
            // The arena base is guaranteed only arena.alignment_bytes.
            // A slot offset divisible by 64 does NOT make a 4-aligned base
            // suitable for a 64-aligned SIMD/native access. This check is
            // immutable plan validation, not a runtime alignment fallback.
            let arena = self.find_arena(slot.arena).ok_or(PlanError::MissingArena)?;
            // Empty slots never form or dereference a physical payload address.
            // Do not reject an otherwise valid zero-byte graph for missing
            // alignment on a pointer that is never materialized.
            if slot.bytes != 0 && !super::super::arena_guarantees_slot_alignment(
                arena.alignment_bytes, slot.alignment_bytes, slot.offset_bytes,
            ) {
                return Err(PlanError::MemorySlotMisaligned);
            }
            if slot.live_from_step > slot.live_until_step {
                return Err(PlanError::InvalidMemoryLifetime);
            }
            if !self.schedule.is_empty() && slot.live_until_step as usize >= self.schedule.len() {
                return Err(PlanError::InvalidMemoryLifetime);
            }
            if slot.lifetime == LifetimeClass::Persistent && !self.schedule.is_empty() {
                let last = self.schedule.len() - 1;
                if slot.live_from_step != 0
                    || usize::try_from(slot.live_until_step).unwrap_or(usize::MAX) != last
                {
                    return Err(PlanError::InvalidMemoryLifetime);
                }
            }
            let arena = self.find_arena(slot.arena).ok_or(PlanError::MissingArena)?;
            if arena.class == MemoryClass::External
                || arena.device_allocation == DeviceAllocationPolicy::External
            {
                // One external pointer owns one zero-offset slot. Metadata-only
                // alias views may add read-only storages on that same slot, but
                // exactly one root storage owns the runtime binding.
                if slot.alias_group != 0 || slot.reuse_group != 0 || slot.offset_bytes != 0 {
                    return Err(PlanError::InvalidMemoryGrouping);
                }
                let mut slots_in_arena = 0usize;
                let mut slot_index = 0usize;
                while slot_index < self.memory_slots.len() {
                    if self.memory_slots[slot_index].arena == arena.id {
                        slots_in_arena = slots_in_arena
                            .checked_add(1)
                            .ok_or(PlanError::InvalidArenaPolicy)?;
                    }
                    slot_index += 1;
                }

                let mut root_storage: Option<&TensorStoragePlan> = None;
                let mut storage_index = 0usize;
                while storage_index < self.storages.len() {
                    let storage = &self.storages[storage_index];
                    if storage.slot == slot.id {
                        if storage.residency != ResidencyPolicy::ExternalBound
                            || storage.physical_bytes != slot.bytes
                        {
                            return Err(PlanError::InvalidArenaPolicy);
                        }
                        if storage.external_alias_of.is_none() {
                            if root_storage.is_some() {
                                return Err(PlanError::InvalidArenaPolicy);
                            }
                            root_storage = Some(storage);
                        }
                    }
                    storage_index += 1;
                }
                let root_storage = root_storage.ok_or(PlanError::InvalidArenaPolicy)?;

                storage_index = 0;
                while storage_index < self.storages.len() {
                    let storage = &self.storages[storage_index];
                    if storage.slot == slot.id && storage.id != root_storage.id
                        && (storage.external_alias_of != Some(root_storage.id)
                            || storage.access != StorageAccess::ReadOnly)
                        {
                            return Err(PlanError::InvalidArenaPolicy);
                        }
                    storage_index += 1;
                }

                if slots_in_arena != 1 || arena.bytes != slot.bytes {
                    return Err(PlanError::InvalidArenaPolicy);
                }
            }
            let end = slot
                .offset_bytes
                .checked_add(slot.bytes)
                .ok_or(PlanError::MemorySlotOutOfBounds)?;
            if end > arena.bytes {
                return Err(PlanError::MemorySlotOutOfBounds);
            }
            i += 1;
        }

        i = 0;
        while i < self.storages.len() {
            let storage = &self.storages[i];
            let tensor = self.find_tensor(storage.tensor).ok_or(PlanError::MissingTensor)?;
            let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
            self.find_arena(slot.arena).ok_or(PlanError::MissingArena)?;
            if let Some(root_id) = storage.external_alias_of {
                let root = self.find_storage(root_id).ok_or(PlanError::MissingStorage)?;
                if root.id == storage.id
                    || root.slot != storage.slot
                    || root.external_alias_of.is_some()
                    || root.residency != ResidencyPolicy::ExternalBound
                    || storage.residency != ResidencyPolicy::ExternalBound
                    || storage.access != StorageAccess::ReadOnly
                {
                    return Err(PlanError::InvalidPhysicalStorage);
                }
            }
            if tensor.layout.alignment_bytes == 0
                || !tensor.layout.alignment_bytes.is_power_of_two()
            {
                return Err(PlanError::InvalidTensorAlignment);
            }
            let logical_bytes = tensor
                .shape
                .byte_count(tensor.dtype)
                .ok_or(PlanError::TensorByteCountOverflow)?;
            if logical_bytes > 0 && storage.physical_bytes == 0 {
                return Err(PlanError::InvalidPhysicalStorage);
            }
            if slot.bytes < storage.physical_bytes
                || storage.base_offset_bytes > storage.physical_bytes
            {
                return Err(PlanError::StorageTooSmall);
            }
            storage.physical_layout.validate_for_dtype(&tensor.shape, tensor.dtype)?;
            if matches!(storage.access, StorageAccess::WriteOnly | StorageAccess::ReadWrite) {
                super::semantic::validate_writable_strided_nonoverlap(
                    &storage.physical_layout,
                    &tensor.shape,
                )?;
            }
            if matches!(
                storage.physical_layout.kind,
                LayoutKind::Blocked | LayoutKind::Packed | LayoutKind::TargetNative
            ) && storage.format_tag == 0
            {
                return Err(PlanError::InvalidPhysicalStorage);
            }
            if let Some(minimum_bytes) = super::semantic::physical_minimum_bytes(
                &storage.physical_layout,
                &tensor.shape,
                tensor.dtype,
                storage.base_offset_bytes,
            )? {
                if storage.physical_bytes < minimum_bytes {
                    return Err(PlanError::StorageTooSmall);
                }
            }
            let storage_address = slot
                .offset_bytes
                .checked_add(storage.base_offset_bytes)
                .ok_or(PlanError::InvalidTensorAlignment)?;
            if slot.alignment_bytes < storage.physical_layout.alignment_bytes as usize
                || storage_address % storage.physical_layout.alignment_bytes as usize != 0
            {
                return Err(PlanError::InvalidTensorAlignment);
            }
            if matches!(
                storage.residency,
                ResidencyPolicy::Persistent | ResidencyPolicy::PersistentWeight
            ) && slot.lifetime != LifetimeClass::Persistent
            {
                return Err(PlanError::InvalidMemoryLifetime);
            }
            if storage.residency == ResidencyPolicy::PersistentWeight
                && (tensor.reuse != ReusePattern::PersistentWeight
                    || storage.access == StorageAccess::WriteOnly)
                {
                    return Err(PlanError::InvalidMemoryLifetime);
                }
            i += 1;
        }

        Ok(())
    }

    pub(super) fn storage_ranges_overlap(
        &self,
        a: StorageId,
        b: StorageId,
    ) -> Result<bool, PlanError> {
        let sa = self.find_storage(a).ok_or(PlanError::MissingStorage)?;
        let sb = self.find_storage(b).ok_or(PlanError::MissingStorage)?;
        let slot_a = self.find_slot(sa.slot).ok_or(PlanError::MissingMemorySlot)?;
        let slot_b = self.find_slot(sb.slot).ok_or(PlanError::MissingMemorySlot)?;
        if slot_a.arena != slot_b.arena {
            return Ok(false);
        }
        let a_end = slot_a
            .offset_bytes
            .checked_add(sa.physical_bytes)
            .ok_or(PlanError::InvalidKernelAssumption)?;
        let b_end = slot_b
            .offset_bytes
            .checked_add(sb.physical_bytes)
            .ok_or(PlanError::InvalidKernelAssumption)?;
        Ok(slot_a.offset_bytes < b_end && slot_b.offset_bytes < a_end)
    }

}

// =============================================================================
// Transfers validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_transfers(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.transfers.len() {
            let transfer = &self.transfers[i];
            // Concurrency is encoded only by TransferBatchPlan and the exact
            // replay schedule; a second overlap-group mechanism would make the
            // plan representation non-canonical.
            if transfer.overlap_group != 0 {
                return Err(PlanError::InvalidTransferBatch);
            }
            let src = self
                .find_storage(transfer.src_storage)
                .ok_or(PlanError::MissingStorage)?;
            let dst = self
                .find_storage(transfer.dst_storage)
                .ok_or(PlanError::MissingStorage)?;
            if src.tensor != transfer.tensor || dst.tensor != transfer.tensor {
                return Err(PlanError::InvalidTransferTensor);
            }

            self.find_tensor(transfer.tensor)
                .ok_or(PlanError::MissingTensor)?;
            if src.format_tag != dst.format_tag {
                return Err(PlanError::TransferFormatMismatch);
            }
            let transferable_bytes = core::cmp::min(src.physical_bytes, dst.physical_bytes);
            let src_end = transfer
                .src_offset_bytes
                .checked_add(transfer.bytes)
                .ok_or(PlanError::InvalidTransferSize)?;
            let dst_end = transfer
                .dst_offset_bytes
                .checked_add(transfer.bytes)
                .ok_or(PlanError::InvalidTransferSize)?;
            let src_slot = self.find_slot(src.slot).ok_or(PlanError::MissingMemorySlot)?;
            let dst_slot = self.find_slot(dst.slot).ok_or(PlanError::MissingMemorySlot)?;
            if transfer.bytes > transferable_bytes
                || src_end > src.physical_bytes
                || dst_end > dst.physical_bytes
                || src_end > src_slot.bytes
                || dst_end > dst_slot.bytes
                || (transfer.chunk_bytes > 0 && transfer.chunk_bytes > transfer.bytes)
            {
                return Err(PlanError::InvalidTransferSize);
            }

            let stream = self
                .find_stream(transfer.stream)
                .ok_or(PlanError::MissingStream)?;
            let src_arena = self
                .arena_of_storage(transfer.src_storage)
                .ok_or(PlanError::MissingArena)?;
            let dst_arena = self
                .arena_of_storage(transfer.dst_storage)
                .ok_or(PlanError::MissingArena)?;

            // Attribute-aware memcpy policy is a frozen copy decision, never a
            // backend default. Non-copy transfer forms must not carry it.
            if let Some(attrs) = transfer.copy_attributes {
                if !matches!(
                    transfer.kind,
                    TransferKind::AsyncPinnedCopy
                        | TransferKind::PageableHostCopy
                        | TransferKind::DeviceToDevice
                        | TransferKind::PeerToPeer
                ) {
                    return Err(PlanError::InvalidMemcpyAttributes);
                }

                let src_is_host = matches!(
                    src_arena.class,
                    MemoryClass::HostAligned
                        | MemoryClass::HostPinned
                        | MemoryClass::HostMappedZeroCopy
                        | MemoryClass::HostPageableDirect
                );
                match attrs.src_access_order {
                    MemcpySrcAccessOrder::Stream => {}
                    MemcpySrcAccessOrder::DuringApiCall | MemcpySrcAccessOrder::Any => {
                        if !src_is_host {
                            return Err(PlanError::InvalidMemcpyAttributes);
                        }
                    }
                }

                if let Some(location) = attrs.src_location_hint {
                    self.validate_memory_location(location)?;
                    let applicable = match src_arena.class {
                        MemoryClass::UnifiedManaged => self.gpu_has_feature(
                            stream.device_ordinal,
                            gpu_feature::CONCURRENT_MANAGED_ACCESS,
                        ),
                        MemoryClass::HostAligned | MemoryClass::HostPageableDirect => self
                            .gpu_has_feature(
                                stream.device_ordinal,
                                gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS,
                            ),
                        _ => false,
                    };
                    if !applicable {
                        return Err(PlanError::InvalidMemcpyAttributes);
                    }
                }
                if let Some(location) = attrs.dst_location_hint {
                    self.validate_memory_location(location)?;
                    let applicable = match dst_arena.class {
                        MemoryClass::UnifiedManaged => self.gpu_has_feature(
                            stream.device_ordinal,
                            gpu_feature::CONCURRENT_MANAGED_ACCESS,
                        ),
                        MemoryClass::HostAligned | MemoryClass::HostPageableDirect => self
                            .gpu_has_feature(
                                stream.device_ordinal,
                                gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS,
                            ),
                        _ => false,
                    };
                    if !applicable {
                        return Err(PlanError::InvalidMemcpyAttributes);
                    }
                }
            }

            let transfer_position = self
                .transfer_schedule_position(transfer.id)
                .ok_or(PlanError::MissingScheduleItem)?;
            if transfer.producer_node.is_none() && transfer.producer_slice.is_some() {
                return Err(PlanError::InvalidDependencyOrder);
            }
            if transfer.consumer_node.is_none() && transfer.consumer_slice.is_some() {
                return Err(PlanError::InvalidDependencyOrder);
            }
            if let Some(producer) = transfer.producer_node {
                let producer_position = self
                    .execution_completion_position(producer, transfer.producer_slice)
                    .ok_or(PlanError::MissingDependencyNode)?;
                if producer_position >= transfer_position {
                    return Err(PlanError::InvalidDependencyOrder);
                }
            }
            if let Some(consumer) = transfer.consumer_node {
                let consumer_position = self
                    .execution_start_position(consumer, transfer.consumer_slice)
                    .ok_or(PlanError::MissingDependencyNode)?;
                if transfer_position >= consumer_position {
                    return Err(PlanError::InvalidDependencyOrder);
                }
            }

            if !matches!(
                transfer.kind,
                TransferKind::UnifiedPrefetch
                    | TransferKind::PageableSystemPrefetch
                    | TransferKind::ManagedDiscardAndPrefetch
            ) && transfer.prefetch_location.is_some()
            {
                return Err(PlanError::InvalidUnifiedPrefetchStorage);
            }

            if transfer.host_boundary != HostBoundaryClass::None {
                if transfer.consumer_node.is_some() || transfer.producer_node.is_none() {
                    return Err(PlanError::InvalidHostBoundary);
                }
                let host_visible_destination = match transfer.kind {
                    TransferKind::AsyncPinnedCopy
                        | TransferKind::PageableHostCopy
                        | TransferKind::MappedPinnedZeroCopy
                        | TransferKind::PageableDirectZeroCopy => matches!(
                        dst_arena.class,
                        MemoryClass::HostAligned
                            | MemoryClass::HostPinned
                            | MemoryClass::HostMappedZeroCopy
                            | MemoryClass::HostPageableDirect
                            | MemoryClass::UnifiedManaged
                    ) || (dst_arena.class == MemoryClass::External && dst_arena.device_ordinal.is_none()),
                    TransferKind::UnifiedPrefetch | TransferKind::PageableSystemPrefetch => matches!(
                        transfer.prefetch_location,
                        Some(MemoryLocation::Host) | Some(MemoryLocation::HostNuma(_))
                    ),
                    _ => false,
                };
                if !host_visible_destination {
                    return Err(PlanError::InvalidHostBoundary);
                }
                if transfer.host_boundary == HostBoundaryClass::ScalarMetadataReadback {
                    let tensor = self.find_tensor(transfer.tensor).ok_or(PlanError::MissingTensor)?;
                    if tensor.shape.element_count() != Some(1)
                        || !matches!(tensor.role, TensorRole::Index | TensorRole::Mask)
                    {
                        return Err(PlanError::InvalidHostBoundary);
                    }
                }
            } else {
                let exits_to_host = transfer.producer_node.is_some()
                    && transfer.consumer_node.is_none()
                    && match transfer.kind {
                        TransferKind::AsyncPinnedCopy
                        | TransferKind::PageableHostCopy
                        | TransferKind::MappedPinnedZeroCopy
                        | TransferKind::PageableDirectZeroCopy => matches!(
                            dst_arena.class,
                            MemoryClass::HostAligned
                                | MemoryClass::HostPinned
                                | MemoryClass::HostMappedZeroCopy
                                | MemoryClass::HostPageableDirect
                                | MemoryClass::UnifiedManaged
                        ) || (dst_arena.class == MemoryClass::External && dst_arena.device_ordinal.is_none()),
                        TransferKind::UnifiedPrefetch | TransferKind::PageableSystemPrefetch => matches!(
                            transfer.prefetch_location,
                            Some(MemoryLocation::Host) | Some(MemoryLocation::HostNuma(_))
                        ),
                        _ => false,
                    };
                if exits_to_host {
                    return Err(PlanError::InvalidHostBoundary);
                }
            }

            match transfer.kind {
                TransferKind::MappedPinnedZeroCopy => {
                    let valid = src_arena.class == MemoryClass::HostMappedZeroCopy
                        || dst_arena.class == MemoryClass::HostMappedZeroCopy;
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                    {
                        return Err(PlanError::InvalidZeroCopyStorage);
                    }
                    if !valid {
                        return Err(PlanError::InvalidZeroCopyStorage);
                    }
                    if !self
                        .gpu_has_feature(stream.device_ordinal, gpu_feature::MAPPED_HOST_MEMORY)
                    {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
                TransferKind::PageableDirectZeroCopy => {
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                    {
                        return Err(PlanError::InvalidZeroCopyStorage);
                    }
                    let pageable = if src_arena.class == MemoryClass::HostPageableDirect {
                        src_arena
                    } else if dst_arena.class == MemoryClass::HostPageableDirect {
                        dst_arena
                    } else {
                        return Err(PlanError::InvalidZeroCopyStorage);
                    };
                    if pageable.device_ordinal != Some(stream.device_ordinal)
                        || !self.gpu_has_feature(
                            stream.device_ordinal,
                            gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS,
                        )
                        || !self.gpu_has_feature(
                            stream.device_ordinal,
                            gpu_feature::PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES,
                        )
                    {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
                TransferKind::PageableHostCopy => {
                    // CUDA pointer copies support ordinary system-pageable host
                    // memory. Keep this as an offline-benchmarked alternative to
                    // pin/register/stage paths, especially for small or one-shot
                    // external buffers where setup/copy overhead can dominate.
                    let src_host = src_arena.class == MemoryClass::HostAligned;
                    let dst_host = dst_arena.class == MemoryClass::HostAligned;
                    let src_device = self.arena_accessible_on_gpu(src_arena, stream.device_ordinal);
                    let dst_device = self.arena_accessible_on_gpu(dst_arena, stream.device_ordinal);
                    if !((src_host && dst_device) || (dst_host && src_device)) {
                        return Err(PlanError::InvalidPageableCopyStorage);
                    }
                }
                TransferKind::AsyncPinnedCopy => {
                    let src_host = matches!(
                        src_arena.class,
                        MemoryClass::HostPinned | MemoryClass::HostMappedZeroCopy
                    );
                    let dst_host = matches!(
                        dst_arena.class,
                        MemoryClass::HostPinned | MemoryClass::HostMappedZeroCopy
                    );
                    let src_device = self.arena_accessible_on_gpu(src_arena, stream.device_ordinal);
                    let dst_device = self.arena_accessible_on_gpu(dst_arena, stream.device_ordinal);
                    if !((src_host && dst_device) || (dst_host && src_device)) {
                        return Err(PlanError::InvalidPinnedCopyStorage);
                    }
                }
                TransferKind::UnifiedPrefetch => {
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                        || src_arena.class != MemoryClass::UnifiedManaged
                        || dst_arena.class != MemoryClass::UnifiedManaged
                    {
                        return Err(PlanError::InvalidUnifiedPrefetchStorage);
                    }
                    if !self.gpu_has_feature(stream.device_ordinal, gpu_feature::UNIFIED_MEMORY) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                    let location = transfer
                        .prefetch_location
                        .ok_or(PlanError::InvalidUnifiedPrefetchStorage)?;
                    self.validate_memory_location(location)?;
                    if let MemoryLocation::Device(device) = location {
                        if !self.gpu_has_feature(device, gpu_feature::CONCURRENT_MANAGED_ACCESS)
                            || !self.gpu_has_feature(
                                stream.device_ordinal,
                                gpu_feature::CONCURRENT_MANAGED_ACCESS,
                            )
                        {
                            return Err(PlanError::GpuFeatureContractMismatch);
                        }
                    }
                }
                TransferKind::PageableSystemPrefetch => {
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                        || src_arena.class != MemoryClass::HostPageableDirect
                        || dst_arena.class != MemoryClass::HostPageableDirect
                    {
                        return Err(PlanError::InvalidUnifiedPrefetchStorage);
                    }
                    if src_arena.device_ordinal != Some(stream.device_ordinal)
                        || !self.gpu_has_feature(
                            stream.device_ordinal,
                            gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS,
                        )
                    {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                    let location = transfer
                        .prefetch_location
                        .ok_or(PlanError::InvalidUnifiedPrefetchStorage)?;
                    self.validate_memory_location(location)?;
                    if let MemoryLocation::Device(device) = location {
                        if device != stream.device_ordinal
                            || !self.gpu_has_feature(device, gpu_feature::CONCURRENT_MANAGED_ACCESS)
                        {
                            return Err(PlanError::GpuFeatureContractMismatch);
                        }
                    }
                }
                TransferKind::ManagedDiscard | TransferKind::ManagedDiscardAndPrefetch => {
                    // CUDA discard is an in-place lifetime/residency operation. It
                    // is only legal for managed memory or system-pageable memory
                    // supported by the driver, and never behaves like a copy.
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                        || transfer.bytes == 0
                    {
                        return Err(PlanError::InvalidManagedDiscardStorage);
                    }
                    if !self.gpu_has_feature(
                        stream.device_ordinal,
                        gpu_feature::BATCHED_MEMORY_DISCARD,
                    ) || !self.all_gpus_have_feature(gpu_feature::CONCURRENT_MANAGED_ACCESS) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }

                    // A discard is only correct when this range is stale and the
                    // very next dataflow owner overwrites it. Freeze that fact
                    // into the plan: the transfer must precede the canonical
                    // producer of this tensor, target the exact GPU output
                    // storage, and share its stream so no replay-time fence or
                    // ownership inference is required.
                    let writer_id = transfer
                        .consumer_node
                        .ok_or(PlanError::InvalidManagedDiscardStorage)?;
                    if transfer.producer_node.is_some()
                        || self.tensor_producer(transfer.tensor) != Some(writer_id)
                        || self
                            .sync_execution_target(writer_id, transfer.consumer_slice)
                            .map_err(|_| PlanError::InvalidManagedDiscardStorage)?
                            != ExecutionTarget::Gpu
                    {
                        return Err(PlanError::InvalidManagedDiscardStorage);
                    }
                    let writer = self
                        .find_node(writer_id)
                        .ok_or(PlanError::InvalidManagedDiscardStorage)?;
                    let writer_gpu = writer
                        .gpu
                        .ok_or(PlanError::InvalidManagedDiscardStorage)?;
                    if writer_gpu.stream != transfer.stream
                        || writer_gpu.device_ordinal != stream.device_ordinal
                        || !writer_gpu.io.output_storages.contains(&transfer.dst_storage)
                    {
                        return Err(PlanError::InvalidManagedDiscardStorage);
                    }

                    match src_arena.class {
                        MemoryClass::UnifiedManaged => {
                            if !self.gpu_has_feature(
                                stream.device_ordinal,
                                gpu_feature::CONCURRENT_MANAGED_ACCESS,
                            ) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        MemoryClass::HostPageableDirect => {
                            if !self.all_gpus_have_feature(
                                gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS,
                            ) {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        _ => return Err(PlanError::InvalidManagedDiscardStorage),
                    }

                    match transfer.kind {
                        TransferKind::ManagedDiscard => {
                            if transfer.prefetch_location.is_some() {
                                return Err(PlanError::InvalidManagedDiscardStorage);
                            }
                        }
                        TransferKind::ManagedDiscardAndPrefetch => {
                            let location = transfer
                                .prefetch_location
                                .ok_or(PlanError::InvalidManagedDiscardStorage)?;
                            self.validate_memory_location(location)?;
                            let MemoryLocation::Device(device) = location else {
                                return Err(PlanError::InvalidManagedDiscardStorage);
                            };
                            if device != writer_gpu.device_ordinal
                                || !self.gpu_has_feature(
                                    device,
                                    gpu_feature::CONCURRENT_MANAGED_ACCESS,
                                )
                                || !self.gpu_has_feature(
                                    stream.device_ordinal,
                                    gpu_feature::CONCURRENT_MANAGED_ACCESS,
                                )
                            {
                                return Err(PlanError::GpuFeatureContractMismatch);
                            }
                        }
                        _ => {}
                    }
                }
                TransferKind::PeerToPeer => {
                    let src_device = src_arena.device_ordinal.ok_or(PlanError::GpuDeviceMismatch)?;
                    let dst_device = dst_arena.device_ordinal.ok_or(PlanError::GpuDeviceMismatch)?;
                    if src_device == dst_device
                        || self.find_gpu(src_device).is_none()
                        || self.find_gpu(dst_device).is_none()
                    {
                        return Err(PlanError::GpuDeviceMismatch);
                    }
                    if !self.gpu_has_feature(src_device, gpu_feature::PEER_TO_PEER)
                        || !self.gpu_has_feature(dst_device, gpu_feature::PEER_TO_PEER)
                    {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                    let peer_allowed = self
                        .canonical
                        .memory_topology
                        .peer_access
                        .iter()
                        .any(|pair| pair.src_device == src_device && pair.dst_device == dst_device);
                    if !peer_allowed
                        || (stream.device_ordinal != src_device && stream.device_ordinal != dst_device)
                    {
                        return Err(PlanError::InvalidMemoryTopology);
                    }
                }
                TransferKind::DeviceToDevice => {
                    if !self.arena_accessible_on_gpu(src_arena, stream.device_ordinal)
                        || !self.arena_accessible_on_gpu(dst_arena, stream.device_ordinal)
                    {
                        return Err(PlanError::GpuDeviceMismatch);
                    }
                }
                TransferKind::DeviceResident | TransferKind::None => {
                    if transfer.src_storage != transfer.dst_storage
                        || transfer.src_offset_bytes != transfer.dst_offset_bytes
                    {
                        return Err(PlanError::InvalidTransferSize);
                    }
                }
            }

            i += 1;
        }

        Ok(())
    }

    pub(super) fn validate_transfer_batches(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.transfer_batches.len() {
            let batch = &self.transfer_batches[i];
            if batch.transfers.is_empty() {
                return Err(PlanError::InvalidTransferBatch);
            }
            let stream = self
                .find_stream(batch.stream)
                .ok_or(PlanError::MissingStream)?;
            match batch.kind {
                TransferBatchKind::AsyncCopy => {
                    if !self.gpu_has_feature(
                        stream.device_ordinal,
                        gpu_feature::BATCHED_ASYNC_COPY,
                    ) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
                TransferBatchKind::MemoryPrefetch => {
                    if !self.gpu_has_feature(
                        stream.device_ordinal,
                        gpu_feature::BATCHED_MANAGED_PREFETCH,
                    ) || !self.all_gpus_have_feature(gpu_feature::CONCURRENT_MANAGED_ACCESS) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
                TransferBatchKind::MemoryDiscard
                | TransferBatchKind::MemoryDiscardAndPrefetch => {
                    if !self.gpu_has_feature(
                        stream.device_ordinal,
                        gpu_feature::BATCHED_MEMORY_DISCARD,
                    ) || !self.all_gpus_have_feature(gpu_feature::CONCURRENT_MANAGED_ACCESS) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
            }

            let mut order_index = 1usize;
            while order_index < batch.transfers.len() {
                if batch.transfers[order_index - 1] >= batch.transfers[order_index] {
                    return Err(PlanError::InvalidTransferBatch);
                }
                order_index += 1;
            }

            let mut j = 0usize;
            while j < batch.transfers.len() {
                let transfer = self
                    .find_transfer(batch.transfers[j])
                    .ok_or(PlanError::InvalidTransferBatch)?;
                let kind_matches_batch = match batch.kind {
                    TransferBatchKind::AsyncCopy => {
                        matches!(
                            transfer.kind,
                            TransferKind::AsyncPinnedCopy
                                | TransferKind::PageableHostCopy
                                | TransferKind::DeviceToDevice
                                | TransferKind::PeerToPeer
                        ) && transfer.copy_attributes.is_some()
                    }
                    TransferBatchKind::MemoryPrefetch => {
                        matches!(
                            transfer.kind,
                            TransferKind::UnifiedPrefetch | TransferKind::PageableSystemPrefetch
                        ) && transfer.copy_attributes.is_none()
                    }
                    TransferBatchKind::MemoryDiscard => {
                        transfer.kind == TransferKind::ManagedDiscard
                            && transfer.copy_attributes.is_none()
                            && transfer.prefetch_location.is_none()
                    }
                    TransferBatchKind::MemoryDiscardAndPrefetch => {
                        transfer.kind == TransferKind::ManagedDiscardAndPrefetch
                            && transfer.copy_attributes.is_none()
                            && transfer.prefetch_location.is_some()
                    }
                };
                if transfer.stream != batch.stream
                    || transfer.host_boundary != HostBoundaryClass::None
                    || !kind_matches_batch
                {
                    return Err(PlanError::InvalidTransferBatch);
                }
                if transfer.kind == TransferKind::PageableSystemPrefetch
                    && !self.all_gpus_have_feature(gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS)
                {
                    return Err(PlanError::GpuFeatureContractMismatch);
                }

                let mut k = j + 1;
                while k < batch.transfers.len() {
                    if batch.transfers[j] == batch.transfers[k] {
                        return Err(PlanError::InvalidTransferBatch);
                    }
                    let other = self
                        .find_transfer(batch.transfers[k])
                        .ok_or(PlanError::InvalidTransferBatch)?;
                    let transfer_src_end = transfer
                        .src_offset_bytes
                        .checked_add(transfer.bytes)
                        .ok_or(PlanError::InvalidTransferBatch)?;
                    let transfer_dst_end = transfer
                        .dst_offset_bytes
                        .checked_add(transfer.bytes)
                        .ok_or(PlanError::InvalidTransferBatch)?;
                    let other_src_end = other
                        .src_offset_bytes
                        .checked_add(other.bytes)
                        .ok_or(PlanError::InvalidTransferBatch)?;
                    let other_dst_end = other
                        .dst_offset_bytes
                        .checked_add(other.bytes)
                        .ok_or(PlanError::InvalidTransferBatch)?;

                    let dst_to_src_overlap = transfer.dst_storage == other.src_storage
                        && transfer.dst_offset_bytes < other_src_end
                        && other.src_offset_bytes < transfer_dst_end;
                    let src_to_dst_overlap = transfer.src_storage == other.dst_storage
                        && transfer.src_offset_bytes < other_dst_end
                        && other.dst_offset_bytes < transfer_src_end;
                    let dst_to_dst_overlap = transfer.dst_storage == other.dst_storage
                        && transfer.dst_offset_bytes < other_dst_end
                        && other.dst_offset_bytes < transfer_dst_end;
                    if dst_to_src_overlap || src_to_dst_overlap || dst_to_dst_overlap {
                        return Err(PlanError::InvalidTransferBatch);
                    }
                    k += 1;
                }

                let mut b = i + 1;
                while b < self.transfer_batches.len() {
                    let mut t = 0usize;
                    while t < self.transfer_batches[b].transfers.len() {
                        if self.transfer_batches[b].transfers[t] == transfer.id {
                            return Err(PlanError::InvalidTransferBatch);
                        }
                        t += 1;
                    }
                    b += 1;
                }
                j += 1;
            }
            i += 1;
        }
        Ok(())
    }

    fn sync_execution_target(
        &self,
        node_id: NodeId,
        slice: Option<u16>,
    ) -> Result<ExecutionTarget, PlanError> {
        let node = self.find_node(node_id).ok_or(PlanError::InvalidSyncReference)?;
        match node.placement {
            Placement::Cpu => {
                if slice.is_some() {
                    return Err(PlanError::InvalidSyncReference);
                }
                Ok(ExecutionTarget::Cpu)
            }
            Placement::Gpu => {
                if slice.is_some() {
                    return Err(PlanError::InvalidSyncReference);
                }
                Ok(ExecutionTarget::Gpu)
            }
            Placement::CpuGpu => {
                let slice_index = slice.ok_or(PlanError::InvalidSyncReference)?;
                let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                mapping
                    .slices
                    .get(usize::from(slice_index))
                    .map(|item| item.target)
                    .ok_or(PlanError::InvalidSyncReference)
            }
        }
    }

    fn sync_gpu_kernel(
        &self,
        node_id: NodeId,
        slice: Option<u16>,
    ) -> Result<&GpuKernelPlan, PlanError> {
        if self.sync_execution_target(node_id, slice)? != ExecutionTarget::Gpu {
            return Err(PlanError::InvalidSyncReference);
        }
        let node = self.find_node(node_id).ok_or(PlanError::InvalidSyncReference)?;
        if node.placement == Placement::CpuGpu {
            let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
            let mut gpu_slices = 0usize;
            let mut i = 0usize;
            while i < mapping.slices.len() {
                if mapping.slices[i].target == ExecutionTarget::Gpu {
                    gpu_slices += 1;
                }
                i += 1;
            }
            // GpuKernelPlan has one completion-event field, not one per slice.
            // Event-based synchronization is exact only for a single GPU slice.
            if gpu_slices != 1 {
                return Err(PlanError::InvalidSyncReference);
            }
        }
        node.gpu.ok_or(PlanError::MissingGpuMapping)
    }

    pub(super) fn validate_sync(&self) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < self.synchronization.len() {
            let sync = &self.synchronization[i];
            if !self.has_node(sync.producer_node) || !self.has_node(sync.consumer_node) {
                return Err(PlanError::InvalidSyncReference);
            }

            let producer_target = self.sync_execution_target(sync.producer_node, sync.producer_slice)?;
            let consumer_target = self.sync_execution_target(sync.consumer_node, sync.consumer_slice)?;

            let sync_position = self
                .sync_schedule_position(sync.id)
                .ok_or(PlanError::MissingScheduleItem)?;
            let producer_position = self
                .execution_completion_position(sync.producer_node, sync.producer_slice)
                .ok_or(PlanError::MissingScheduleItem)?;
            let consumer_position = self
                .execution_start_position(sync.consumer_node, sync.consumer_slice)
                .ok_or(PlanError::MissingScheduleItem)?;
            if producer_position >= sync_position || sync_position >= consumer_position {
                return Err(PlanError::InvalidDependencyOrder);
            }

            match sync.kind {
                SyncKind::StreamWaitEvent => {
                    if producer_target != ExecutionTarget::Gpu
                        || consumer_target != ExecutionTarget::Gpu
                        || sync.barrier.is_some()
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                    let event_id = sync.event.ok_or(PlanError::MissingEvent)?;
                    let stream_id = sync.stream.ok_or(PlanError::MissingStream)?;
                    let producer = self.sync_gpu_kernel(sync.producer_node, sync.producer_slice)?;
                    let consumer = self.sync_gpu_kernel(sync.consumer_node, sync.consumer_slice)?;
                    let event = self.find_event(event_id).ok_or(PlanError::MissingEvent)?;
                    let stream = self.find_stream(stream_id).ok_or(PlanError::MissingStream)?;
                    if producer.launch_completion_event != Some(event_id)
                        || consumer.stream != stream_id
                        || producer.device_ordinal != consumer.device_ordinal
                        || event.device_ordinal != producer.device_ordinal
                        || stream.device_ordinal != consumer.device_ordinal
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                }
                SyncKind::ProgrammaticDependency => {
                    if producer_target != ExecutionTarget::Gpu
                        || consumer_target != ExecutionTarget::Gpu
                        || sync.event.is_some()
                        || sync.barrier.is_some()
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                    let stream_id = sync.stream.ok_or(PlanError::MissingStream)?;
                    let producer = self.sync_gpu_kernel(sync.producer_node, sync.producer_slice)?;
                    let consumer = self.sync_gpu_kernel(sync.consumer_node, sync.consumer_slice)?;
                    let stream = self.find_stream(stream_id).ok_or(PlanError::MissingStream)?;
                    if producer.device_ordinal != consumer.device_ordinal
                        || producer.execution_context != consumer.execution_context
                        || producer.stream != stream_id
                        || consumer.stream != stream_id
                        || stream.device_ordinal != producer.device_ordinal
                        || stream.execution_context != producer.execution_context
                        || !producer.programmatic_dependent_launch
                        || !consumer.programmatic_dependent_launch
                        || !consumer.programmatic_stream_serialization
                        || !self.gpu_has_feature(
                            producer.device_ordinal,
                            gpu_feature::PROGRAMMATIC_DEPENDENT_LAUNCH,
                        )
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                }
                SyncKind::CpuWaitGpuEvent => {
                    if producer_target != ExecutionTarget::Gpu
                        || consumer_target != ExecutionTarget::Cpu
                        || sync.stream.is_some()
                        || sync.barrier.is_some()
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                    let event_id = sync.event.ok_or(PlanError::MissingEvent)?;
                    let producer = self.sync_gpu_kernel(sync.producer_node, sync.producer_slice)?;
                    let event = self.find_event(event_id).ok_or(PlanError::MissingEvent)?;
                    if producer.launch_completion_event != Some(event_id)
                        || event.device_ordinal != producer.device_ordinal
                    {
                        return Err(PlanError::InvalidSyncReference);
                    }
                }
                // These forms do not yet carry enough frozen information for an
                // exact dependency in this IR (barrier participants, external
                // event provenance, or an explicit record+wait pair). They are
                // reserved rather than delegated to backend interpretation.
                SyncKind::None
                | SyncKind::CpuBarrier
                | SyncKind::GpuEventRecord
                | SyncKind::GpuWaitExternal => {
                    return Err(PlanError::InvalidSyncReference);
                }
            }
            i += 1;
        }
        Ok(())
    }

}

// =============================================================================
// Schedule validation
// =============================================================================

impl FrozenExecutionPlan {
    pub(super) fn validate_schedule(&self) -> Result<(), PlanError> {
        let mut unbatched_transfers = 0usize;
        let mut t = 0usize;
        while t < self.transfers.len() {
            if self.batch_for_transfer(self.transfers[t].id).is_none() {
                unbatched_transfers = unbatched_transfers
                    .checked_add(1)
                    .ok_or(PlanError::MissingScheduleItem)?;
            }
            t += 1;
        }

        // A single-target node occupies one execution item. A combined CPU-GPU
        // node occupies one explicit launch and one explicit completion item per
        // frozen slice, so overlap/sequencing is represented rather than inferred.
        let mut execution_items = 0usize;
        let mut n = 0usize;
        while n < self.nodes.len() {
            let node = &self.nodes[n];
            if node.placement == Placement::CpuGpu {
                let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                let slice_items = mapping
                    .slices
                    .len()
                    .checked_mul(2)
                    .ok_or(PlanError::MissingScheduleItem)?;
                execution_items = execution_items
                    .checked_add(slice_items)
                    .ok_or(PlanError::MissingScheduleItem)?;
            } else {
                execution_items = execution_items
                    .checked_add(1)
                    .ok_or(PlanError::MissingScheduleItem)?;
            }
            n += 1;
        }

        let expected = execution_items
            .checked_add(unbatched_transfers)
            .and_then(|v| v.checked_add(self.transfer_batches.len()))
            .and_then(|v| v.checked_add(self.synchronization.len()))
            .ok_or(PlanError::MissingScheduleItem)?;
        if self.schedule.len() != expected {
            return Err(PlanError::MissingScheduleItem);
        }

        let mut i = 0usize;
        while i < self.schedule.len() {
            match self.schedule[i] {
                ScheduleItem::Node(id) => {
                    let node = self
                        .nodes
                        .iter()
                        .find(|node| node.id == id)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    if node.placement == Placement::CpuGpu {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                }
                ScheduleItem::CpuGpuSliceLaunch { node, slice }
                | ScheduleItem::CpuGpuSliceComplete { node, slice } => {
                    let frozen_node = self
                        .nodes
                        .iter()
                        .find(|candidate| candidate.id == node)
                        .ok_or(PlanError::MissingScheduleItem)?;
                    if frozen_node.placement != Placement::CpuGpu {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    let mapping = frozen_node
                        .cpu_gpu
                        .ok_or(PlanError::InvalidCpuGpuMapping)?;
                    if usize::from(slice) >= mapping.slices.len() {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                }
                ScheduleItem::Transfer(id) => {
                    if !self.has_transfer(id) || self.batch_for_transfer(id).is_some() {
                        return Err(PlanError::InvalidTransferBatch);
                    }
                }
                ScheduleItem::TransferBatch(id) => {
                    if !self.has_transfer_batch(id) {
                        return Err(PlanError::MissingScheduleItem);
                    }
                }
                ScheduleItem::Sync(id) => {
                    if !self.has_sync(id) {
                        return Err(PlanError::MissingScheduleItem);
                    }
                }
            }

            let mut j = i + 1;
            while j < self.schedule.len() {
                if self.schedule[i] == self.schedule[j] {
                    return Err(PlanError::DuplicateScheduleItem);
                }
                j += 1;
            }
            i += 1;
        }

        // Validate exact execution representation for every node.
        n = 0;
        while n < self.nodes.len() {
            let node = &self.nodes[n];
            let start = self
                .node_schedule_start_position(node.id)
                .ok_or(PlanError::MissingScheduleItem)?;
            let completion = self
                .node_schedule_completion_position(node.id)
                .ok_or(PlanError::MissingScheduleItem)?;
            if start > completion {
                return Err(PlanError::InvalidDependencyOrder);
            }

            if node.placement == Placement::CpuGpu {
                let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                let mut slice_index = 0usize;
                let mut previous_completion: Option<usize> = None;
                while slice_index < mapping.slices.len() {
                    if slice_index > u16::MAX as usize {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    let slice_u16 = slice_index as u16;
                    let launch_item = ScheduleItem::CpuGpuSliceLaunch {
                        node: node.id,
                        slice: slice_u16,
                    };
                    let completion_item = ScheduleItem::CpuGpuSliceComplete {
                        node: node.id,
                        slice: slice_u16,
                    };
                    let mut launch_position = None;
                    let mut completion_position = None;
                    let mut schedule_index = 0usize;
                    while schedule_index < self.schedule.len() {
                        if self.schedule[schedule_index] == launch_item {
                            launch_position = Some(schedule_index);
                        }
                        if self.schedule[schedule_index] == completion_item {
                            completion_position = Some(schedule_index);
                        }
                        schedule_index += 1;
                    }
                    let launch_position = launch_position.ok_or(PlanError::MissingScheduleItem)?;
                    let completion_position =
                        completion_position.ok_or(PlanError::MissingScheduleItem)?;
                    if launch_position >= completion_position {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                    if !mapping.overlap_cpu_gpu {
                        if let Some(previous) = previous_completion {
                            if previous >= launch_position {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                        }
                        previous_completion = Some(completion_position);
                    }
                    slice_index += 1;
                }

                // Pipeline dependencies are exact per work slice. A stage-N slice
                // may launch only after the matching stage-(N-1) slice completed.
                // Different work slices may still overlap across CPU and GPU, which
                // gives a real frozen pipeline instead of an inferred backend policy.
                if mapping.mode == CpuGpuMode::Pipeline {
                    let mut current_index = 0usize;
                    while current_index < mapping.slices.len() {
                        let current = mapping.slices[current_index];
                        if current.stage > 0 {
                            let mut predecessor: Option<usize> = None;
                            let mut candidate_index = 0usize;
                            while candidate_index < mapping.slices.len() {
                                let candidate = mapping.slices[candidate_index];
                                if candidate.stage + 1 == current.stage
                                    && candidate.work == current.work
                                {
                                    if predecessor.is_some() {
                                        return Err(PlanError::InvalidCpuGpuMapping);
                                    }
                                    predecessor = Some(candidate_index);
                                }
                                candidate_index += 1;
                            }
                            let predecessor =
                                predecessor.ok_or(PlanError::InvalidCpuGpuMapping)?;
                            if predecessor > u16::MAX as usize || current_index > u16::MAX as usize {
                                return Err(PlanError::InvalidCpuGpuMapping);
                            }
                            let predecessor_completion = self
                                .cpu_gpu_slice_completion_position(node.id, predecessor as u16)
                                .ok_or(PlanError::MissingScheduleItem)?;
                            let current_launch = self
                                .cpu_gpu_slice_launch_position(node.id, current_index as u16)
                                .ok_or(PlanError::MissingScheduleItem)?;
                            if predecessor_completion >= current_launch {
                                return Err(PlanError::InvalidDependencyOrder);
                            }
                        }
                        current_index += 1;
                    }
                }

                // The overlap flag describes the actual frozen schedule. If it is
                // enabled, at least one CPU slice and one GPU slice must be in flight
                // at the same time; otherwise the plan would contain advisory metadata.
                if mapping.overlap_cpu_gpu {
                    let mut has_cross_target_overlap = false;
                    let mut a = 0usize;
                    while a < mapping.slices.len() && !has_cross_target_overlap {
                        let mut b = a + 1;
                        while b < mapping.slices.len() && !has_cross_target_overlap {
                            if mapping.slices[a].target != mapping.slices[b].target {
                                if a > u16::MAX as usize || b > u16::MAX as usize {
                                    return Err(PlanError::InvalidCpuGpuMapping);
                                }
                                let a_launch = self
                                    .cpu_gpu_slice_launch_position(node.id, a as u16)
                                    .ok_or(PlanError::MissingScheduleItem)?;
                                let a_complete = self
                                    .cpu_gpu_slice_completion_position(node.id, a as u16)
                                    .ok_or(PlanError::MissingScheduleItem)?;
                                let b_launch = self
                                    .cpu_gpu_slice_launch_position(node.id, b as u16)
                                    .ok_or(PlanError::MissingScheduleItem)?;
                                let b_complete = self
                                    .cpu_gpu_slice_completion_position(node.id, b as u16)
                                    .ok_or(PlanError::MissingScheduleItem)?;
                                has_cross_target_overlap =
                                    a_launch < b_complete && b_launch < a_complete;
                            }
                            b += 1;
                        }
                        a += 1;
                    }
                    if !has_cross_target_overlap {
                        return Err(PlanError::InvalidCpuGpuMapping);
                    }
                }
            } else if start != completion {
                return Err(PlanError::MissingScheduleItem);
            }
            n += 1;
        }

        i = 0;
        while i < self.transfers.len() {
            if self.transfer_schedule_position(self.transfers[i].id).is_none() {
                return Err(PlanError::MissingScheduleItem);
            }
            i += 1;
        }

        i = 0;
        while i < self.transfer_batches.len() {
            let id = self.transfer_batches[i].id;
            let mut found = false;
            let mut j = 0usize;
            while j < self.schedule.len() {
                if self.schedule[j] == ScheduleItem::TransferBatch(id) {
                    found = true;
                    break;
                }
                j += 1;
            }
            if !found {
                return Err(PlanError::MissingScheduleItem);
            }
            i += 1;
        }

        i = 0;
        while i < self.synchronization.len() {
            if self.sync_schedule_position(self.synchronization[i].id).is_none() {
                return Err(PlanError::MissingScheduleItem);
            }
            i += 1;
        }

        Ok(())
    }

    /// Validate graph-level submission segments. These segments are immutable
    /// contiguous portions of the ordinary schedule, so CUDA Graph replay can
    /// replace repeated host-side per-node submission without changing semantics.
    pub(super) fn validate_gpu_graph_segments(&self) -> Result<(), PlanError> {
        if self.gpu_graph_segments.is_empty() {
            return Ok(());
        }

        let mut previous_end = 0u32;
        let mut i = 0usize;
        while i < self.gpu_graph_segments.len() {
            let segment = self.gpu_graph_segments[i];
            let hw = self
                .find_gpu(segment.device_ordinal)
                .ok_or(PlanError::GpuDeviceMismatch)?;
            if segment.id as usize != i
                || segment.mode == GraphReplayMode::None
                || segment.schedule_begin >= segment.schedule_end
                || segment.schedule_end as usize > self.schedule.len()
                || (i > 0 && segment.schedule_begin < previous_end)
                || segment.device_ordinal != hw.device_ordinal
                || !hw.features.contains(gpu_feature::CUDA_GRAPH)
            {
                return Err(PlanError::InvalidGpuGraphSegment);
            }
            match segment.mode {
                GraphReplayMode::None => return Err(PlanError::InvalidGpuGraphSegment),
                GraphReplayMode::HostGraph => {}
                GraphReplayMode::DeviceTailLaunch
                | GraphReplayMode::DeviceFireAndForget
                | GraphReplayMode::DeviceSiblingLaunch => {
                    if !hw.features.contains(gpu_feature::DEVICE_GRAPH_LAUNCH) {
                        return Err(PlanError::GpuFeatureContractMismatch);
                    }
                }
            }
            let context = self
                .find_context(segment.execution_context)
                .ok_or(PlanError::MissingExecutionContext)?;
            if context.device_ordinal != segment.device_ordinal {
                return Err(PlanError::InvalidGpuGraphSegment);
            }
            let launch_stream = self
                .find_stream(segment.launch_stream)
                .ok_or(PlanError::MissingStream)?;
            if launch_stream.device_ordinal != segment.device_ordinal
                || launch_stream.execution_context != segment.execution_context
            {
                return Err(PlanError::InvalidGpuGraphSegment);
            }
            if let Some(stream_id) = segment.preupload_stream {
                let stream = self.find_stream(stream_id).ok_or(PlanError::MissingStream)?;
                if stream.device_ordinal != segment.device_ordinal
                    || stream.execution_context != segment.execution_context
                {
                    return Err(PlanError::InvalidGpuGraphSegment);
                }
            }
            if matches!(
                segment.mode,
                GraphReplayMode::DeviceTailLaunch
                    | GraphReplayMode::DeviceFireAndForget
                    | GraphReplayMode::DeviceSiblingLaunch
            ) && (!segment.single_instantiation || segment.preupload_stream.is_none())
            {
                return Err(PlanError::InvalidGpuGraphSegment);
            }

            // A segment containing only metadata aliases (or synchronization)
            // would pay graph-launch overhead without submitting useful device
            // work. Such aliases remain in the ordinary schedule for lifetime
            // accounting, but replay elides them entirely.
            let mut has_executable_work = false;
            let mut position = segment.schedule_begin as usize;
            while position < segment.schedule_end as usize {
                match self.schedule[position] {
                    ScheduleItem::Node(node_id) => {
                        let node = self
                            .nodes
                            .get(node_id as usize)
                            .filter(|item| item.id == node_id)
                            .ok_or(PlanError::MissingScheduleItem)?;
                        if node.placement != Placement::Gpu {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        let gpu = node.gpu.ok_or(PlanError::MissingGpuMapping)?;
                        if gpu.device_ordinal != segment.device_ordinal
                            || gpu.execution_context != segment.execution_context
                            || gpu.graph_replay != GraphReplayMode::None
                        {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        if gpu.launch_priority.is_some() && !segment.use_node_priority {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        if gpu.family != KernelFamily::AliasView {
                            has_executable_work = true;
                        }
                    }
                    ScheduleItem::Transfer(id) => {
                        let transfer = self.find_transfer(id).ok_or(PlanError::MissingScheduleItem)?;
                        if transfer.host_boundary != HostBoundaryClass::None {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        let stream = self.find_stream(transfer.stream).ok_or(PlanError::MissingStream)?;
                        if stream.device_ordinal != segment.device_ordinal
                            || stream.execution_context != segment.execution_context
                        {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        has_executable_work = true;
                    }
                    ScheduleItem::TransferBatch(id) => {
                        let batch = self
                            .find_transfer_batch(id)
                            .ok_or(PlanError::MissingScheduleItem)?;
                        let stream = self.find_stream(batch.stream).ok_or(PlanError::MissingStream)?;
                        if stream.device_ordinal != segment.device_ordinal
                            || stream.execution_context != segment.execution_context
                        {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        let mut t = 0usize;
                        while t < batch.transfers.len() {
                            let transfer = self
                                .find_transfer(batch.transfers[t])
                                .ok_or(PlanError::MissingScheduleItem)?;
                            if transfer.host_boundary != HostBoundaryClass::None {
                                return Err(PlanError::InvalidGpuGraphSegment);
                            }
                            t += 1;
                        }
                        if !batch.transfers.is_empty() {
                            has_executable_work = true;
                        }
                    }
                    ScheduleItem::Sync(id) => {
                        let sync = self
                            .synchronization
                            .get(id as usize)
                            .filter(|item| item.id == id)
                            .ok_or(PlanError::MissingScheduleItem)?;
                        if matches!(
                            sync.kind,
                            SyncKind::CpuBarrier | SyncKind::CpuWaitGpuEvent | SyncKind::GpuWaitExternal
                        ) {
                            return Err(PlanError::InvalidGpuGraphSegment);
                        }
                        if let Some(stream_id) = sync.stream {
                            let stream = self.find_stream(stream_id).ok_or(PlanError::MissingStream)?;
                            if stream.device_ordinal != segment.device_ordinal
                                || stream.execution_context != segment.execution_context
                            {
                                return Err(PlanError::InvalidGpuGraphSegment);
                            }
                        }
                        if sync.kind == SyncKind::ProgrammaticDependency {
                            let producer_position = self
                                .execution_completion_position(sync.producer_node, sync.producer_slice)
                                .ok_or(PlanError::MissingScheduleItem)?;
                            let consumer_position = self
                                .execution_start_position(sync.consumer_node, sync.consumer_slice)
                                .ok_or(PlanError::MissingScheduleItem)?;
                            if producer_position < segment.schedule_begin as usize
                                || consumer_position >= segment.schedule_end as usize
                            {
                                return Err(PlanError::InvalidGpuGraphSegment);
                            }
                        }
                    }
                    ScheduleItem::CpuGpuSliceLaunch { .. }
                    | ScheduleItem::CpuGpuSliceComplete { .. } => {
                        return Err(PlanError::InvalidGpuGraphSegment);
                    }
                }
                position += 1;
            }
            if !has_executable_work {
                return Err(PlanError::InvalidGpuGraphSegment);
            }
            previous_end = segment.schedule_end;
            i += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod external_arena_device_membership_tests {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tests/tensor_light/unit/execution/validation/external_arena_device_membership_tests.rs"
    ));
}
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/exact_empty_batched_matmul.rs"]
mod exact_empty_batched_matmul;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/degenerate_axis_and_single_input.rs"]
mod degenerate_axis_and_single_input;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/zero_padding_semantics.rs"]
mod zero_padding_semantics;
// v58: these identities are SEMANTICALLY VALID even though any physical
// alias/copy/zero-insertion still requires a released exact FrozenExecutionPlan.
// These are Rust source tests until cargo test is executed on a real toolchain.
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/zero_and_identity_shape_contracts.rs"]
mod zero_and_identity_shape_contracts;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/scalar_tile_and_single_input_join.rs"]
mod scalar_tile_and_single_input_join;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/writable_stride_injectivity.rs"]
mod writable_stride_injectivity;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/validation/empty_boolean_and_count_reductions.rs"]
mod empty_boolean_and_count_reductions;
