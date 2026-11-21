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
// Subcomponent:  Runtime
// Module:        momiru::tensor_light::execution::runtime
// Source:        src/tensor_light/execution/runtime.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Deterministic frozen-plan resolver, resource binder and schedule replayer.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Resolve exact admitted execution plans and target capabilities.
//   - Validate resource bindings before entering replay.
//   - Replay the frozen plan without altering operation semantics or schedules.
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

//! Thin deterministic runtime: exact canonical-plan resolution, one-time
//! capability/resource binding, then central replay of the frozen schedule.

use super::ir::{CanonicalWorkload, MemoryTopologyContract};
use std::sync::Arc;
use super::{PlanError, StorageId};
use crate::tensor_light::tensor::DType;
use super::plan::{
    CpuGpuSlice, CpuKernelPlan, DeviceAllocationPolicy, ExecutionTarget, FrozenExecutionPlan,
    FrozenNode, GpuGraphSegmentPlan, GpuKernelPlan, HostBoundaryClass, KernelFamily, MemoryClass, Placement, ResidencyPolicy,
    ScheduleItem, StorageAccess, SyncKind, SyncPlan, TensorStoragePlan, TransferBatchPlan, TransferKind, TransferPlan,
};
use super::target::{gpu_feature, HardwareContract};

// =============================================================================
// Exact plan database
// =============================================================================

/// A one-time, checked index over an already authenticated frozen catalog.
/// Key indexing only narrows exact matches; canonical structural equality
/// remains mandatory. The index never selects a candidate or changes a plan.
pub struct PlanDb {
    plans: &'static [FrozenExecutionPlan],
    /// (signature_hi, signature_lo, original plan offset), sorted at init.
    key_index: Vec<(u64, u64, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanResolveError {
    NoExactPlan,
    AmbiguousExactPlan,
}

/// Locate the *complete* matching signature bucket, including hash collisions.
/// The input is sorted once while constructing PlanDb. No allocation or
/// optimization decision occurs during exact resolution.
#[inline]
fn exact_key_window(index: &[(u64, u64, usize)], needle: (u64, u64))
    -> core::ops::Range<usize>
{
    let lower = index.partition_point(|entry| (entry.0, entry.1) < needle);
    let upper = lower + index[lower..]
        .partition_point(|entry| (entry.0, entry.1) == needle);
    lower..upper
}

/// Reject identical published plans while leaving genuine digest collisions
/// with DIFFERENT canonical identities admissible. This is strictly one-time
/// catalog validation: there is no workload, cost, or plan-choice arbitration.
/// The index is already sorted by its two digests, and the supplied predicate
/// compares complete canonical identities only inside one collision bucket.
/// A quadratic scan is bounded by bucket size, which is normally one.
fn duplicate_exact_identity<F>(
    index: &[(u64, u64, usize)], mut identical: F,
) -> bool
where
    F: FnMut(usize, usize) -> bool,
{
    let mut begin = 0usize;
    while begin < index.len() {
        let signature = (index[begin].0, index[begin].1);
        let end = begin + index[begin..]
            .partition_point(|entry| (entry.0, entry.1) == signature);
        for current in begin + 1..end {
            for previous in begin..current {
                if identical(index[previous].2, index[current].2) {
                    return true;
                }
            }
        }
        begin = end;
    }
    false
}

impl PlanDb {
    /// Register already linked, trusted, compile-time materialized frozen plans.
    ///
    /// Every plan is validated and indexed exactly once at construction. This
    /// API accepts Rust-static plans compiled into the same binary; it is not
    /// an authenticated loader for arbitrary plan bytes received from disk or
    /// a network. Such a loader needs a separate cryptographic trust boundary.
    /// No backend kernel or runtime compilation is added by registration.
    #[inline]
    pub fn from_static_plans(plans: &'static [FrozenExecutionPlan]) -> Result<Self, PlanError> {
        Self::new(plans)
    }

    /// Internal only: a safe public constructor must authenticate the signed
    /// catalog first. All immutable plan certificates are checked ONCE before
    /// they become resolvable. Index construction fails closed on invalid
    /// plans or allocation failure, never falling back to an unverified scan.
    /// Per-submission physical bindings and capabilities remain independently
    /// checked at the exact binding boundary (they are not immutable).
    pub(crate) fn new(plans: &'static [FrozenExecutionPlan]) -> Result<Self, PlanError> {
        let mut key_index = Vec::new();
        key_index.try_reserve_exact(plans.len())
            .map_err(|_| PlanError::PlanLookupIndexAllocationFailed)?;
        for (index, plan) in plans.iter().enumerate() {
            // FrozenExecutionPlan::validate is immutable structural/evidence
            // validation; it must not be repeated for every warm binding.
            plan.validate()?;
            key_index.push((plan.key.signature_hi, plan.key.signature_lo, index));
        }
        // In-place, non-allocating sort. Equal keys retain their entire
        // bucket, and canonical records are checked before accepting a hit.
        key_index.sort_unstable_by_key(|entry| (entry.0, entry.1));
        // Enforce the catalog's exact uniqueness invariant at initialization,
        // not during a later caller's hot lookup. Never mistake a digest
        // collision between DIFFERENT canonical graphs for a duplicate.
        if duplicate_exact_identity(&key_index, |left, right| {
            let a = &plans[left];
            let b = &plans[right];
            a.key == b.key && a.workload == b.workload && a.canonical == b.canonical
        }) {
            return Err(PlanError::DuplicateFrozenPlanIdentity);
        }
        Ok(Self { plans, key_index })
    }

    /// Collision-safe exact resolution from the complete canonical workload.
    /// Every return is a structurally validated plan: initialization has
    /// rejected the entire catalog if any plan certificate is invalid.
    /// No fallback, priority, heuristic or performance arbitration is allowed.
    /// Index lookup: O(log N + K), K = same-signature collision count;
    /// structural match cost depends on the canonical graph size.
    pub fn resolve_workload_exact(
        &self,
        workload: &CanonicalWorkload,
    ) -> Result<&'static FrozenExecutionPlan, PlanResolveError> {
        let signature = workload.signature();
        let key = signature.plan_key();
        let range = exact_key_window(
            &self.key_index, (key.signature_hi, key.signature_lo),
        );
        let mut found: Option<&'static FrozenExecutionPlan> = None;
        for &(_, _, original_index) in &self.key_index[range] {
            let plan = &self.plans[original_index];
            if plan.key == key
                && plan.workload == signature
                && plan.canonical == *workload
            {
                if found.is_some() {
                    return Err(PlanResolveError::AmbiguousExactPlan);
                }
                found = Some(plan);
            }
        }
        found.ok_or(PlanResolveError::NoExactPlan)
    }
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/exact_catalog_key_window_tests.rs"]
mod exact_catalog_key_window_tests;
impl core::fmt::Display for PlanResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PlanResolveError {}

// =============================================================================
// Runtime resource boundary
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostBufferHandle {
    pub address: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceBufferHandle {
    pub device_ordinal: u16,
    pub address: u64,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferHandle {
    Host(HostBufferHandle),
    HostPinned(HostBufferHandle),
    HostMappedZeroCopy {
        host: HostBufferHandle,
        device_address: u64,
        device_ordinal: u16,
    },
    Device(DeviceBufferHandle),
    Unified {
        host_address: usize,
        device_address: u64,
        bytes: usize,
        device_ordinal: u16,
    },
    External {
        host_address: Option<usize>,
        device_address: Option<u64>,
        bytes: usize,
        device_ordinal: Option<u16>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingAccess {
    ReadOnly,
    ReadWrite,
}

mod binding_element_sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u64 {}
}

/// Host element types that can be bound to canonical tensor storage.
/// Sealing prevents external code from claiming an incorrect dtype.
pub trait BindingElement: binding_element_sealed::Sealed + Copy + 'static {
    const DTYPE: DType;
}

impl BindingElement for f32 { const DTYPE: DType = DType::F32; }
impl BindingElement for i32 { const DTYPE: DType = DType::I32; }
impl BindingElement for i64 { const DTYPE: DType = DType::I64; }
impl BindingElement for u8 { const DTYPE: DType = DType::U8; }
impl BindingElement for u64 { const DTYPE: DType = DType::U64; }

/// One external storage binding. Its lifetime ties every safe host binding or
/// owner-tied raw handle to the complete bound execution that retains it.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StorageBinding<'a> {
    storage: StorageId,
    buffer: BufferHandle,
    access: BindingAccess,
    dtype: DType,
    _source_lifetime: core::marker::PhantomData<&'a ()>,
}

impl<'a> StorageBinding<'a> {
    #[inline]
    pub fn host_slice<T: BindingElement>(storage: StorageId, data: &'a [T]) -> Self {
        Self {
            storage,
            buffer: BufferHandle::Host(HostBufferHandle {
                address: data.as_ptr().expose_provenance(),
                bytes: core::mem::size_of_val(data),
            }),
            access: BindingAccess::ReadOnly,
            dtype: T::DTYPE,
            _source_lifetime: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn host_slice_mut<T: BindingElement>(storage: StorageId, data: &'a mut [T]) -> Self {
        Self {
            storage,
            buffer: BufferHandle::Host(HostBufferHandle {
                address: data.as_mut_ptr().expose_provenance(),
                bytes: core::mem::size_of_val(data),
            }),
            access: BindingAccess::ReadWrite,
            dtype: T::DTYPE,
            _source_lifetime: core::marker::PhantomData,
        }
    }

    /// Bind an externally owned raw handle.
    ///
    /// # Safety
    /// The handle must remain valid, correctly aligned, and contain elements of
    /// `dtype` for the supplied owner's lifetime. Its addresses must refer to
    /// the allocation represented by that owner.
    #[inline]
    pub unsafe fn from_raw_read_only<Owner: ?Sized>(
        storage: StorageId,
        buffer: BufferHandle,
        dtype: DType,
        _owner: &'a Owner,
    ) -> Self {
        Self {
            storage,
            buffer,
            access: BindingAccess::ReadOnly,
            dtype,
            _source_lifetime: core::marker::PhantomData,
        }
    }

    /// Writable raw-handle form.
    ///
    /// # Safety
    /// The handle must satisfy the read-only requirements above and must be
    /// writable for the complete owner lifetime. The caller must not expose a
    /// second alias that can violate the frozen plan's access contract.
    #[inline]
    pub unsafe fn from_raw_read_write<Owner: ?Sized>(
        storage: StorageId,
        buffer: BufferHandle,
        dtype: DType,
        _owner: &'a mut Owner,
    ) -> Self {
        Self {
            storage,
            buffer,
            access: BindingAccess::ReadWrite,
            dtype,
            _source_lifetime: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub const fn storage(&self) -> StorageId {
        self.storage
    }

    #[inline(always)]
    pub const fn buffer(&self) -> BufferHandle {
        self.buffer
    }

    #[inline(always)]
    pub const fn access(&self) -> BindingAccess {
        self.access
    }

    #[inline(always)]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }
}

impl BufferHandle {
    #[inline(always)]
    const fn bytes(self) -> usize {
        match self {
            Self::Host(handle) | Self::HostPinned(handle) => handle.bytes,
            Self::HostMappedZeroCopy { host, .. } => host.bytes,
            Self::Device(handle) => handle.bytes,
            Self::Unified { bytes, .. } | Self::External { bytes, .. } => bytes,
        }
    }
}

/// Check integer-address representability before a raw external descriptor can
/// be bound to a frozen native kernel. This cannot prove real allocation bounds
/// or provenance: those still belong to the unsafe owner/backend contract.
///
/// A nonempty host range is [address, address + bytes), with a representable
/// exclusive end and at most isize::MAX bytes as required for Rust slice-like
/// pointer arithmetic. Empty ranges require no formed or dereferenced pointer.
#[inline(always)]
fn host_byte_range_representable(address: usize, bytes: usize) -> bool {
    bytes == 0 || (address != 0
        && bytes <= isize::MAX as usize
        && address.checked_add(bytes).is_some())
}

/// Device virtual addresses use the exact u64 native-address ABI of this
/// limited CUDA/HIP-style binding form; they are NOT host Rust pointers.
#[inline(always)]
fn device_byte_range_representable(address: u64, bytes: usize) -> bool {
    if bytes == 0 { return true; }
    let Ok(native_bytes) = u64::try_from(bytes) else { return false; };
    address != 0 && address.checked_add(native_bytes).is_some()
}

/// Reject overflow of either address domain for mapped/unified descriptors.
/// Even with no dereference at this boundary, passing an overflowing range
/// into a native launch ABI would invalidate its certified byte-span guard.
#[inline]
fn buffer_address_ranges_representable(buffer: BufferHandle) -> bool {
    match buffer {
        BufferHandle::Host(h) | BufferHandle::HostPinned(h) => {
            host_byte_range_representable(h.address, h.bytes)
        }
        BufferHandle::Device(h) => device_byte_range_representable(h.address, h.bytes),
        BufferHandle::HostMappedZeroCopy { host, device_address, .. } => {
            host_byte_range_representable(host.address, host.bytes)
                && device_byte_range_representable(device_address, host.bytes)
        }
        BufferHandle::Unified { host_address, device_address, bytes, .. } => {
            host_byte_range_representable(host_address, bytes)
                && device_byte_range_representable(device_address, bytes)
        }
        BufferHandle::External { host_address, device_address, bytes, .. } => {
            host_address.is_none_or(|addr| host_byte_range_representable(addr, bytes))
                && device_address.is_none_or(|addr| device_byte_range_representable(addr, bytes))
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/checked_external_address_range_tests.rs"]
mod checked_external_address_range_tests;
#[inline(always)]
fn byte_ranges_overlap(a_start: u128, a_bytes: usize, b_start: u128, b_bytes: usize) -> bool {
    if a_bytes == 0 || b_bytes == 0 {
        return false;
    }
    let a_end = a_start + a_bytes as u128;
    let b_end = b_start + b_bytes as u128;
    a_start < b_end && b_start < a_end
}

/// A real *proof of disjointness*, not simply failure to observe an overlap.
/// The v26 light binding ABI has NO allocation generation, device context,
/// alias-owner identity, or managed-memory coherence certificate. Therefore
/// mapped/managed/external and peer-device writable pairs are UNKNOWN and
/// rejected rather than being accepted by comparing unrelated virtual ranges.
/// Host-only bindings are slices governed by Rust borrowing; their validated
/// host address ranges may be compared within the same process address space.
#[inline]
fn external_buffers_proven_disjoint(a: BufferHandle, b: BufferHandle) -> bool {
    if a.bytes() == 0 || b.bytes() == 0 { return true; }
    match (a, b) {
        (
            BufferHandle::Host(a) | BufferHandle::HostPinned(a),
            BufferHandle::Host(b) | BufferHandle::HostPinned(b),
        ) => !byte_ranges_overlap(a.address as u128, a.bytes, b.address as u128, b.bytes),
        // All other pairs lack sufficient common physical allocation identity
        // for a safe writable no-alias proof in this still-limited ABI.
        _ => false,
    }
}

/// Required base-pointer alignment for an externally substituted arena.
///
/// The storage descriptor, slot AND arena each carry a minimum alignment
/// guarantee. A weakly aligned external pointer cannot substitute for a
/// strongly aligned arena even when a relative slot offset is aligned.
#[inline(always)]
fn external_base_alignment(storage: usize, slot: usize, arena: usize) -> usize {
    storage.max(slot).max(arena)
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/external_base_alignment_tests.rs"]
mod external_base_alignment_tests;
/// A foreign base pointer replaces the *whole* immutable arena, not just
/// the logical tensor payload of one storage. A root slot may begin at a
/// nonzero byte offset, and the same arena may contain additional frozen
/// slots. Checking only `storage.physical_bytes` is insufficient to prove
/// that the native address range covers the prevalidated resource layout.
///
/// The caller must provide the entire arena capacity in its descriptor;
/// requiring the same frozen size for every bind is not a runtime allocator
/// choice or a plan fallback. Native allocation provenance is still the
/// backend/caller's explicit unsafe obligation.
#[inline(always)]
fn external_binding_covers_arena(provided_bytes: usize, arena_bytes: usize) -> bool {
    provided_bytes >= arena_bytes
}

/// A foreign device address is valid only for an ordinal admitted by the
/// *frozen* target topology. A device advertised elsewhere in the hardware
/// inventory is not automatically part of this workload's memory mapping.
/// When the arena freezes an owning device, the foreign handle must name it.
///
/// This is a necessary bind-time guard; allocation provenance, CUDA context,
/// pointer registration and native last-use leases remain backend obligations.
#[inline]
fn external_device_matches_frozen_topology(
    bound_ordinal: u16,
    frozen_ordinals: &[u16],
    arena_device_ordinal: Option<u16>,
) -> bool {
    frozen_ordinals.contains(&bound_ordinal)
        && arena_device_ordinal.is_none_or(|required| required == bound_ordinal)
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/external_device_mapping_tests.rs"]
mod external_device_mapping_tests;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/external_arena_coverage_tests.rs"]
mod external_arena_coverage_tests;
impl FrozenExecutionPlan {
    fn storage_requires_external_binding(
        &self,
        storage: &TensorStoragePlan,
    ) -> Result<bool, PlanError> {
        if storage.external_alias_of.is_some() {
            return Ok(false);
        }
        let arena = self.arena_of_storage(storage.id).ok_or(PlanError::MissingArena)?;
        Ok(storage.residency == ResidencyPolicy::ExternalBound
            || arena.class == MemoryClass::External
            || arena.device_allocation == DeviceAllocationPolicy::External)
    }

    pub fn validate_external_bindings(&self, bindings: &[StorageBinding<'_>]) -> Result<(), PlanError> {
        let mut i = 0usize;
        while i < bindings.len() {
            let binding = &bindings[i];
            let storage = self
                .find_storage(binding.storage)
                .ok_or(PlanError::UnexpectedExternalBinding)?;
            if !self.storage_requires_external_binding(storage)? {
                return Err(PlanError::UnexpectedExternalBinding);
            }
            let tensor = self
                .tensors
                .iter()
                .find(|tensor| tensor.id == storage.tensor)
                .ok_or(PlanError::MissingTensor)?;
            if binding.dtype != tensor.dtype {
                return Err(PlanError::InvalidExternalBinding);
            }
            if matches!(storage.access, StorageAccess::WriteOnly | StorageAccess::ReadWrite)
                && binding.access != BindingAccess::ReadWrite
            {
                return Err(PlanError::InvalidExternalBinding);
            }

            let mut j = i + 1;
            while j < bindings.len() {
                if bindings[j].storage == binding.storage {
                    return Err(PlanError::DuplicateExternalBinding);
                }
                j += 1;
            }

            let arena = self.arena_of_storage(storage.id).ok_or(PlanError::MissingArena)?;
            // The bound address is the arena base. The original storage-only
            // length test admitted undersized foreign allocations whenever
            // a frozen slot had a nonzero offset or the arena had other slots.
            // Immutable plan validation already proves each storage lies
            // within its arena; bind must verify the COMPLETE foreign range.
            if !external_binding_covers_arena(binding.buffer.bytes(), arena.bytes)
                || !buffer_address_ranges_representable(binding.buffer)
            {
                return Err(PlanError::InvalidExternalBinding);
            }
            let slot = self.find_slot(storage.slot).ok_or(PlanError::MissingMemorySlot)?;
            // The real external pointer substitutes for an arena base. It
            // must satisfy ALL three frozen guarantees, including the arena
            // base alignment (which may exceed the slot/kernel alignment).
            let required_alignment = external_base_alignment(
                storage.physical_layout.alignment_bytes as usize,
                slot.alignment_bytes,
                arena.alignment_bytes,
            );
            let nonempty = storage.physical_bytes != 0;

            let aligned_host = |address: usize| -> bool {
                !nonempty || (address != 0 && address.is_multiple_of(required_alignment))
            };
            let aligned_device = |address: u64| -> bool {
                !nonempty || (address != 0 && address.is_multiple_of(required_alignment as u64))
            };

            let valid = match (arena.class, binding.buffer) {
                (MemoryClass::HostAligned, BufferHandle::Host(handle)) => aligned_host(handle.address),
                (MemoryClass::HostPageableDirect, BufferHandle::Host(handle)) => {
                    aligned_host(handle.address)
                        && arena.device_ordinal.is_some_and(|device| {
                            self.find_gpu(device)
                                .map(|gpu| {
                                    gpu.features.contains(gpu_feature::PAGEABLE_HOST_MEMORY_ACCESS)
                                        && gpu.features.contains(
                                            gpu_feature::PAGEABLE_HOST_MEMORY_USES_HOST_PAGE_TABLES,
                                        )
                                })
                                .unwrap_or(false)
                        })
                }
                (MemoryClass::HostPinned, BufferHandle::HostPinned(handle)) => aligned_host(handle.address),
                (
                    MemoryClass::HostMappedZeroCopy,
                    BufferHandle::HostMappedZeroCopy { host, device_address, device_ordinal },
                ) => {
                    aligned_host(host.address)
                        && aligned_device(device_address)
                        // A mapped address belongs to the frozen MEMORY
                        // topology. This set can include memory-only GPUs
                        // omitted from the execution-placement GPU subset.
                        // Checking execution_topology here rejected a valid
                        // offline mapping; accepting an arbitrary hardware
                        // GPU would be equally incorrect. No remapping occurs.
                        && external_device_matches_frozen_topology(
                            device_ordinal,
                            self.canonical.memory_topology.gpu_device_ordinals,
                            arena.device_ordinal,
                        )
                        && self
                            .find_gpu(device_ordinal)
                            .map(|gpu| gpu.features.contains(gpu_feature::MAPPED_HOST_MEMORY))
                            .unwrap_or(false)
                }
                (MemoryClass::DeviceLocal, BufferHandle::Device(handle)) => {
                    arena.device_ordinal == Some(handle.device_ordinal) && aligned_device(handle.address)
                }
                (
                    MemoryClass::UnifiedManaged,
                    BufferHandle::Unified { host_address, device_address, device_ordinal, .. },
                ) => {
                    // A managed allocation can have a designated owner
                    // even when its CPU access is separate from GPU execution.
                    // Validate membership in the sealed memory topology and
                    // exact arena affinity; a mere hardware feature bit is
                    // not enough to authorize a different GPU's address.
                    external_device_matches_frozen_topology(
                        device_ordinal,
                        self.canonical.memory_topology.gpu_device_ordinals,
                        arena.device_ordinal,
                    )
                        && self
                            .find_gpu(device_ordinal)
                            .map(|gpu| gpu.features.contains(gpu_feature::UNIFIED_MEMORY))
                            .unwrap_or(false)
                        && aligned_host(host_address)
                        && aligned_device(device_address)
                }
                (
                    MemoryClass::External,
                    BufferHandle::External { host_address, device_address, device_ordinal, .. },
                ) => {
                    let addresses_aligned = host_address.map(aligned_host).unwrap_or(true)
                        && device_address.map(aligned_device).unwrap_or(true);
                    let location_matches = match arena.device_ordinal {
                        None => {
                            host_address.is_some()
                                && device_address.is_none()
                                && device_ordinal.is_none()
                        }
                        Some(expected_device) => {
                            host_address.is_none()
                                && device_address.is_some()
                                && device_ordinal == Some(expected_device)
                        }
                    };
                    location_matches && addresses_aligned
                }
                _ => false,
            };
            if !valid {
                return Err(PlanError::InvalidExternalBinding);
            }
            i += 1;
        }

        // This light plan format has no explicit external-arena base/alias
        // identity. Static validation therefore forbids alias/reuse groups on
        // externally bound arenas, and binding rejects writable physical
        // overlap OR UNKNOWN disjointness. Mere different device addresses do
        // not prove non-aliasing. Two retained read-only leases can coexist.
        i = 0;
        while i < bindings.len() {
            let mut j = i + 1;
            while j < bindings.len() {
                let a = &bindings[i];
                let b = &bindings[j];
                if (a.access == BindingAccess::ReadWrite || b.access == BindingAccess::ReadWrite)
                    && !external_buffers_proven_disjoint(a.buffer, b.buffer)
                {
                    return Err(PlanError::InvalidExternalBinding);
                }
                j += 1;
            }
            i += 1;
        }

        i = 0;
        while i < self.storages.len() {
            let storage = &self.storages[i];
            if self.storage_requires_external_binding(storage)? {
                let mut found = false;
                let mut j = 0usize;
                while j < bindings.len() {
                    if bindings[j].storage == storage.id {
                        found = true;
                        break;
                    }
                    j += 1;
                }
                if !found {
                    return Err(PlanError::MissingExternalBinding);
                }
            }
            i += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeCapabilityBinding {
    /// Exact detected capability payload. Hashes below are seals/lookup aids;
    /// strict binding compares these full immutable contracts as well.
    pub hardware: HardwareContract,
    pub memory_topology: MemoryTopologyContract,
    pub hardware_profile_hash_hi: u64,
    pub hardware_profile_hash_lo: u64,
    pub memory_topology_hash_hi: u64,
    pub memory_topology_hash_lo: u64,
}

impl RuntimeCapabilityBinding {
    /// Build the exact one-time capability identity from detected hardware and
    /// memory/interconnect topology. This performs no performance selection.
    #[inline]
    pub fn canonical(
        hardware: &HardwareContract,
        memory_topology: &MemoryTopologyContract,
    ) -> Self {
        let hardware_hash = hardware.computed_profile_hash();
        let memory_hash = memory_topology.canonical_hash();
        Self {
            hardware: *hardware,
            memory_topology: *memory_topology,
            hardware_profile_hash_hi: hardware_hash.0,
            hardware_profile_hash_lo: hardware_hash.1,
            memory_topology_hash_hi: memory_hash.0,
            memory_topology_hash_lo: memory_hash.1,
        }
    }

    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        (self.hardware_profile_hash_hi | self.hardware_profile_hash_lo) != 0
            && (self.memory_topology_hash_hi | self.memory_topology_hash_lo) != 0
    }

    #[inline(always)]
    pub fn matches_plan(self, plan: &FrozenExecutionPlan) -> bool {
        self.hardware == plan.hardware
            && self.hardware == plan.canonical.hardware
            && self.memory_topology == plan.canonical.memory_topology
            && self.hardware_profile_hash_hi == plan.hardware.profile_hash_hi
            && self.hardware_profile_hash_lo == plan.hardware.profile_hash_lo
            && self.hardware_profile_hash_hi == plan.workload.hardware_hi
            && self.hardware_profile_hash_lo == plan.workload.hardware_lo
            && self.memory_topology_hash_hi == plan.workload.memory_topology_hi
            && self.memory_topology_hash_lo == plan.workload.memory_topology_lo
    }
}

// =============================================================================
// Pre-bound primitives and central replay
// =============================================================================

/// Backend object after one-time binding. It exposes only exact execution
/// primitives. It never receives authority to choose schedule, placement,
/// layout, kernel family or candidate.
/// # Safety
/// Implementors MUST preserve all Rust host-borrow and native buffer lifetimes.
/// Each dispatch can enqueue asynchronous native work, but `quiesce_or_abort`
/// must establish every last-user completion/coherence edge before returning.
/// It MUST terminate nonreturning when safe quiescence cannot be proven.
/// A restricted synchronous built-in f32 CPU backend is supplied in
/// `execution::cpu`; complete GPU, hybrid and external-native backends are not.
pub unsafe trait BoundBackend {
    type Error;

    fn execute_cpu_node(
        &mut self,
        node: &'static FrozenNode,
        kernel: &'static CpuKernelPlan,
    ) -> Result<(), Self::Error>;

    fn execute_gpu_node(
        &mut self,
        node: &'static FrozenNode,
        kernel: &'static GpuKernelPlan,
    ) -> Result<(), Self::Error>;

    fn launch_cpu_slice(
        &mut self,
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
        kernel: &'static CpuKernelPlan,
    ) -> Result<(), Self::Error>;

    fn launch_gpu_slice(
        &mut self,
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
        kernel: &'static GpuKernelPlan,
    ) -> Result<(), Self::Error>;

    fn complete_cpu_gpu_slice(
        &mut self,
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
    ) -> Result<(), Self::Error>;

    /// Launch a graph executable that was instantiated/pre-uploaded during
    /// one-time binding from the exact frozen schedule segment.
    fn execute_gpu_graph_segment(
        &mut self,
        segment: &'static GpuGraphSegmentPlan,
    ) -> Result<(), Self::Error>;

    fn execute_transfer(&mut self, transfer: &'static TransferPlan) -> Result<(), Self::Error>;

    /// Host-visible readback primitive. It must return only after the exact
    /// transfer is visible to the host; no implicit global/context synchronize
    /// or runtime policy decision is permitted.
    fn execute_host_readback(
        &mut self,
        transfer: &'static TransferPlan,
        boundary: HostBoundaryClass,
    ) -> Result<(), Self::Error>;
    /// Submit one already-bound native transfer batch. The backend must build
    /// and retain any pointer/size/location descriptor arrays during one-time
    /// binding; replay performs exactly one batch submission and no per-member
    /// dispatch, allocation, or descriptor construction.
    fn execute_transfer_batch(
        &mut self,
        batch: &'static TransferBatchPlan,
    ) -> Result<(), Self::Error>;
    fn execute_sync(&mut self, sync: &'static SyncPlan) -> Result<(), Self::Error>;

    /// MUST NOT unwind. On failure to prove native/CPU-worker quiescence,
    /// abort or remain nonreturning, never release a live borrowed host slice.
    fn quiesce_or_abort(&mut self);
}

/// # Safety
/// `bind_resources` MUST NOT enqueue or retain in-flight reads/writes of
/// borrowed external memory before `BoundExecution::replay`. The returned
/// BoundBackend must obey its unsafe last-use/quiescence contract and must not
/// create hidden runtime kernels or alternative schedules.
pub unsafe trait Backend {
    type Bound: BoundBackend<Error = Self::Error>;
    type Error;

    /// One-time exact hardware plus memory/interconnect capability identity.
    /// This is capability binding only; no performance search or arbitration is allowed.
    fn bind_capabilities() -> Result<RuntimeCapabilityBinding, Self::Error>;

    /// One-time allocation/resource/artifact binding according to the frozen
    /// plan. External bindings are supplied only for root storages; read-only
    /// storages with `external_alias_of` derive their address/base from that root.
    /// This phase owns all replay-invariant setup: arena allocation and
    /// prefaulting, NUMA placement, persistent CPU worker-team creation, native
    /// artifact/module loading, tensor-map encoding, stream/event construction,
    /// memory-pool setup, native transfer-batch descriptor/attribute arrays, and
    /// No CUDA-graph instantiation: GPU graph segments are rejected by
    /// strict validation until a pre-existing native-executable proof exists.
    /// Replay must not
    /// lazily create any of those resources. Substitution of kernels, layouts,
    /// placement or schedule is invalid.
    /// # Safety
    /// The caller must provide a fully validated immutable frozen plan and
    /// bindings already checked by `FrozenExecutionPlan::validate_external_bindings`,
    /// and retain *all original external storage owners* and their
    /// shared/exclusive borrows for the entire lifetime of the returned
    /// bound backend, including the last native use. Calling this method
    /// directly without such a lifetime witness is unsound for backends that
    /// prebind raw host/device pointers. Prefer `PreparedExecution::bind`,
    /// which retains the original bindings and establishes this guarantee.
    unsafe fn bind_resources(
        plan: &'static FrozenExecutionPlan,
        capabilities: RuntimeCapabilityBinding,
        external_bindings: &[StorageBinding<'_>],
    ) -> Result<Self::Bound, Self::Error>;
}

#[derive(Clone, Copy)]
enum ReplayStep {
    CpuNode {
        node: &'static FrozenNode,
        kernel: &'static CpuKernelPlan,
    },
    GpuNode {
        node: &'static FrozenNode,
        kernel: &'static GpuKernelPlan,
    },
    CpuSlice {
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
        kernel: &'static CpuKernelPlan,
    },
    GpuSlice {
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
        kernel: &'static GpuKernelPlan,
    },
    CpuGpuComplete {
        node: &'static FrozenNode,
        slice_index: u16,
        slice: &'static CpuGpuSlice,
    },
    GpuGraphSegment(&'static GpuGraphSegmentPlan),
    Transfer(&'static TransferPlan),
    HostReadback { transfer: &'static TransferPlan, boundary: HostBoundaryClass },
    TransferBatch(&'static TransferBatchPlan),
    Sync(&'static SyncPlan),
}

/// An unconditional last-use fence, including if a backend returns Err or
/// unwinds. The unsafe BoundBackend contract forbids this method from
/// unwinding; it must abort/nonreturn if quiescence is not provable.
/// This private guard cannot be forgotten or handed to safe library users.
struct ReplayQuiescenceGuard<'a, B: BoundBackend> {
    backend: &'a mut B,
}

impl<'a, B: BoundBackend> Drop for ReplayQuiescenceGuard<'a, B> {
    #[inline(always)]
    fn drop(&mut self) {
        self.backend.quiesce_or_abort();
    }
}

pub struct BoundExecution<'a, B: BoundBackend> {
    plan: &'static FrozenExecutionPlan,
    replay_steps: Arc<Vec<ReplayStep>>,
    backend: B,
    // Retaining the bindings pins source/owner lifetimes for every replay.
    // Retain the *original* Vec allocation and its lifetime witnesses.
    // Vec::into_boxed_slice could shrink/reallocate AFTER native binding.
    // No copy, new arena or allocator action is permitted at that boundary.
    _external_bindings: Vec<StorageBinding<'a>>,
}

impl<'a, B: BoundBackend> BoundExecution<'a, B> {
    #[inline(always)]
    pub const fn plan(&self) -> &'static FrozenExecutionPlan {
        self.plan
    }

    /// Replay the exact prelowered steps. The RAII guard is intentionally
    /// stack-local and private: Rust drops it on success, early backend error,
    /// or unwind, synchronizing all in-flight borrowed-memory users BEFORE
    /// the caller can access their input/output slices again.
    /// No per-replay catch_unwind closure/panic payload dispatch is required.
    pub fn replay(&mut self) -> Result<(), B::Error> {
        let guard = ReplayQuiescenceGuard { backend: &mut self.backend };
        Self::dispatch_frozen_steps(&mut *guard.backend, &self.replay_steps)
    }

    /// No resource search or backend dispatch arbitration is performed here.
    #[inline]
    fn dispatch_frozen_steps(backend: &mut B, steps: &[ReplayStep]) -> Result<(), B::Error> {
        for step in steps {
            match step {
                ReplayStep::CpuNode { node, kernel } => {
                    backend.execute_cpu_node(node, kernel)?;
                }
                ReplayStep::GpuNode { node, kernel } => {
                    backend.execute_gpu_node(node, kernel)?;
                }
                ReplayStep::CpuSlice { node, slice_index, slice, kernel } => {
                    backend.launch_cpu_slice(node, *slice_index, slice, kernel)?;
                }
                ReplayStep::GpuSlice { node, slice_index, slice, kernel } => {
                    backend.launch_gpu_slice(node, *slice_index, slice, kernel)?;
                }
                ReplayStep::CpuGpuComplete { node, slice_index, slice } => {
                    backend.complete_cpu_gpu_slice(node, *slice_index, slice)?;
                }
                ReplayStep::GpuGraphSegment(segment) => {
                    backend.execute_gpu_graph_segment(segment)?;
                }
                ReplayStep::Transfer(transfer) => {
                    backend.execute_transfer(transfer)?;
                }
                ReplayStep::HostReadback { transfer, boundary } => {
                    backend.execute_host_readback(transfer, *boundary)?;
                }
                ReplayStep::TransferBatch(batch) => {
                    backend.execute_transfer_batch(batch)?;
                }
                ReplayStep::Sync(sync) => {
                    backend.execute_sync(sync)?;
                }
            }
        }
        Ok(())
    }
}

/// Allocate the exact immutable replay-stage budget at bind time, returning a
/// resource error instead of relying on infallible Vec allocation.
/// No later push can exceed this pre-reserved schedule-length bound.
fn reserve_replay_steps(bound: usize) -> Result<Vec<ReplayStep>, PlanError> {
    let mut steps = Vec::new();
    steps.try_reserve_exact(bound)
        .map_err(|_| PlanError::ReplayProgramAllocationFailed)?;
    Ok(steps)
}

fn build_replay_steps(plan: &'static FrozenExecutionPlan) -> Result<Vec<ReplayStep>, PlanError> {
    // A frozen transfer batch is one native submission, not a replay-time loop.
    // CUDA memcpy/prefetch batch descriptors are materialized once by
    // `Backend::bind_resources`, so schedule length is already a conservative
    // upper bound for the compact replay program.
    let mut steps = reserve_replay_steps(plan.schedule.len())?;

    let mut i = 0usize;
    let mut graph_segment_index = 0usize;
    while i < plan.schedule.len() {
        if graph_segment_index < plan.gpu_graph_segments.len() {
            let segment = &plan.gpu_graph_segments[graph_segment_index];
            if i == segment.schedule_begin as usize {
                steps.push(ReplayStep::GpuGraphSegment(segment));
                i = segment.schedule_end as usize;
                graph_segment_index += 1;
                continue;
            }
        }
        match plan.schedule[i] {
            ScheduleItem::Node(id) => {
                let node = find_node(plan, id).ok_or(PlanError::MissingScheduleItem)?;
                // AliasView is pure metadata: storage, base offset and strides
                // were frozen and validated before binding. Emitting a CPU call
                // or GPU launch here would turn a zero-copy transform into work.
                if node_is_metadata_only_alias(node) {
                    i += 1;
                    continue;
                }
                match node.placement {
                    Placement::Cpu => steps.push(ReplayStep::CpuNode {
                        node,
                        kernel: node.cpu.ok_or(PlanError::MissingCpuMapping)?,
                    }),
                    Placement::Gpu => steps.push(ReplayStep::GpuNode {
                        node,
                        kernel: node.gpu.ok_or(PlanError::MissingGpuMapping)?,
                    }),
                    Placement::CpuGpu => return Err(PlanError::InvalidCpuGpuMapping),
                }
            }
            ScheduleItem::CpuGpuSliceLaunch { node: node_id, slice } => {
                let node = find_node(plan, node_id).ok_or(PlanError::MissingScheduleItem)?;
                let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                let slice_plan = mapping.slices.get(usize::from(slice)).ok_or(PlanError::InvalidCpuGpuMapping)?;
                match slice_plan.target {
                    ExecutionTarget::Cpu => steps.push(ReplayStep::CpuSlice {
                        node,
                        slice_index: slice,
                        slice: slice_plan,
                        kernel: node.cpu.ok_or(PlanError::MissingCpuMapping)?,
                    }),
                    ExecutionTarget::Gpu => steps.push(ReplayStep::GpuSlice {
                        node,
                        slice_index: slice,
                        slice: slice_plan,
                        kernel: node.gpu.ok_or(PlanError::MissingGpuMapping)?,
                    }),
                }
            }
            ScheduleItem::CpuGpuSliceComplete { node: node_id, slice } => {
                let node = find_node(plan, node_id).ok_or(PlanError::MissingScheduleItem)?;
                let mapping = node.cpu_gpu.ok_or(PlanError::InvalidCpuGpuMapping)?;
                let slice_plan = mapping.slices.get(usize::from(slice)).ok_or(PlanError::InvalidCpuGpuMapping)?;
                steps.push(ReplayStep::CpuGpuComplete { node, slice_index: slice, slice: slice_plan });
            }
            ScheduleItem::Transfer(id) => {
                let transfer = find_transfer(plan, id).ok_or(PlanError::MissingScheduleItem)?;
                if transfer.host_boundary == HostBoundaryClass::None {
                    // No-movement and zero-copy markers affect storage/lifetime
                    // semantics only. They are validated to preserve the same
                    // storage+offset and therefore require no replay backend call.
                    if !matches!(
                        transfer.kind,
                        TransferKind::None
                            | TransferKind::DeviceResident
                            | TransferKind::MappedPinnedZeroCopy
                            | TransferKind::PageableDirectZeroCopy
                    ) {
                        steps.push(ReplayStep::Transfer(transfer));
                    }
                } else {
                    // Host-boundary markers are retained because the backend must
                    // provide the frozen completion/visibility guarantee.
                    steps.push(ReplayStep::HostReadback {
                        transfer,
                        boundary: transfer.host_boundary,
                    });
                }
            }
            ScheduleItem::TransferBatch(id) => {
                let batch = find_transfer_batch(plan, id).ok_or(PlanError::MissingScheduleItem)?;
                steps.push(ReplayStep::TransferBatch(batch));
            }
            ScheduleItem::Sync(id) => {
                let sync = find_sync(plan, id).ok_or(PlanError::MissingScheduleItem)?;
                // PDL is encoded in launch attributes (or a CUDA-graph
                // programmatic dependency) during one-time binding. A separate
                // host-side wait here would destroy the intended launch overlap.
                if sync.kind != SyncKind::ProgrammaticDependency {
                    steps.push(ReplayStep::Sync(sync));
                }
            }
        }
        i += 1;
    }
    Ok(steps)
}


#[inline(always)]
fn node_is_metadata_only_alias(node: &FrozenNode) -> bool {
    match node.placement {
        Placement::Cpu => node.cpu.is_some_and(|kernel| kernel.family == KernelFamily::AliasView),
        Placement::Gpu => node.gpu.is_some_and(|kernel| kernel.family == KernelFamily::AliasView),
        Placement::CpuGpu => false,
    }
}

fn find_node(plan: &'static FrozenExecutionPlan, id: u32) -> Option<&'static FrozenNode> {
    plan.nodes.get(id as usize).filter(|item| item.id == id)
}

fn find_transfer(plan: &'static FrozenExecutionPlan, id: u32) -> Option<&'static TransferPlan> {
    plan.transfers.get(id as usize).filter(|item| item.id == id)
}

fn find_transfer_batch(plan: &'static FrozenExecutionPlan, id: u32) -> Option<&'static TransferBatchPlan> {
    plan.transfer_batches.get(id as usize).filter(|item| item.id == id)
}

fn find_sync(plan: &'static FrozenExecutionPlan, id: u32) -> Option<&'static SyncPlan> {
    plan.synchronization.get(id as usize).filter(|item| item.id == id)
}

#[derive(Debug)]
pub enum BindError<E> {
    NoExactPlan,
    AmbiguousExactPlan,
    InvalidPlan(PlanError),
    /// Host memory could not be reserved for the prebound fixed replay program.
    InsufficientWorkspace,
    Backend(E),
}

impl<E: core::fmt::Display> core::fmt::Display for BindError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoExactPlan => f.write_str("no exact frozen execution plan exists for the canonical workload"),
            Self::AmbiguousExactPlan => f.write_str("more than one exact frozen execution plan matches the canonical workload"),
            Self::InvalidPlan(error) => write!(f, "invalid frozen execution plan or binding: {error}"),
            Self::InsufficientWorkspace => f.write_str("insufficient host workspace for the prebound frozen replay program"),
            Self::Backend(error) => write!(f, "backend binding/execution setup failed: {error}"),
        }
    }
}

impl<E> std::error::Error for BindError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPlan(error) => Some(error),
            Self::Backend(error) => Some(error),
            Self::NoExactPlan | Self::AmbiguousExactPlan | Self::InsufficientWorkspace => None,
        }
    }
}

/// One-time capability binder for repeated exact-plan bindings.
///
/// This is not a second runtime or a placement-specific execution layer. It is
/// only the immutable result of `Backend::bind_capabilities`, cached once so a
/// process that binds several frozen workloads does not re-detect the same CPU,
/// GPU and memory/interconnect topology. It performs no tuning or selection.
pub struct ExecutionBinder<B: Backend> {
    capabilities: RuntimeCapabilityBinding,
    _backend: core::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> ExecutionBinder<B> {
    /// Detect and validate the exact runtime capability identity once.
    pub fn bind_once() -> Result<Self, BindError<B::Error>> {
        let capabilities = B::bind_capabilities().map_err(BindError::Backend)?;
        if !capabilities.is_valid() {
            return Err(BindError::InvalidPlan(PlanError::InvalidHardwareContract));
        }
        Ok(Self {
            capabilities,
            _backend: core::marker::PhantomData,
        })
    }

    #[inline(always)]
    pub const fn capabilities(&self) -> RuntimeCapabilityBinding {
        self.capabilities
    }

    /// Validate and resolve the complete immutable workload exactly once for
    /// repeated submissions. The result contains a single chosen frozen plan;
    /// no kernel/placement tuning or runtime schedule selection occurs.
    #[inline]
    pub fn prepare_workload_exact(
        &self,
        db: &PlanDb,
        workload: &CanonicalWorkload,
    ) -> Result<PreparedExecution<B>, BindError<B::Error>> {
        prepare_workload_exact_with_capabilities::<B>(db, workload, self.capabilities)
    }

    /// Bind one canonical workload using the already detected capability
    /// identity. Resolution remains exact and collision-safe; no fallback,
    /// cost-model arbitration, autotuning or placement/schedule search occurs.
    #[inline]
    pub fn bind_workload_exact<'a>(
        &self,
        db: &PlanDb,
        workload: &CanonicalWorkload,
        external_bindings: Vec<StorageBinding<'a>>,
    ) -> Result<BoundExecution<'a, B::Bound>, BindError<B::Error>> {
        bind_workload_exact_with_capabilities::<B>(
            db,
            workload,
            self.capabilities,
            external_bindings,
        )
    }
}

/// Convenience entry point for callers that bind a single workload. For
/// repeated bindings prefer [`ExecutionBinder::bind_once`] so capability
/// detection is truly one-time for the process/backend instance.
pub fn bind_workload_exact<'a, B: Backend>(
    db: &PlanDb,
    workload: &CanonicalWorkload,
    external_bindings: Vec<StorageBinding<'a>>,
) -> Result<BoundExecution<'a, B::Bound>, BindError<B::Error>> {
    let binder = ExecutionBinder::<B>::bind_once()?;
    binder.bind_workload_exact(db, workload, external_bindings)
}

/// Pre-resolved immutable executable identity. Does not own device resources.
/// Plan resolution, workload validation and target capability checking happen
/// only in prepare_workload_exact. The schedule is lowered to dispatch steps
/// just once; subsequent bind() calls reuse the same immutable replay program.
pub struct PreparedExecution<B: Backend> {
    plan: &'static FrozenExecutionPlan,
    capabilities: RuntimeCapabilityBinding,
    replay_steps: Arc<Vec<ReplayStep>>,
    _backend_type: core::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> PreparedExecution<B> {
    #[inline(always)]
    pub const fn plan(&self) -> &'static FrozenExecutionPlan {
        self.plan
    }

    #[inline(always)]
    pub const fn capabilities(&self) -> RuntimeCapabilityBinding {
        self.capabilities
    }

    /// Bind exact physical buffers and native resources for this one plan.
    /// Checks are repeated for each invocation because addresses and alias
    /// permissions can change. This method does not choose another plan.
    pub fn bind<'a>(
        &self,
        external_bindings: Vec<StorageBinding<'a>>,
    ) -> Result<BoundExecution<'a, B::Bound>, BindError<B::Error>> {
        self.plan.validate_external_bindings(&external_bindings)
            .map_err(BindError::InvalidPlan)?;
        // SAFETY: the owned Vec of lifetime-witness bindings is retained in
        // BoundExecution until AFTER the backend is dropped. Backend binding
        // must not start async work; ReplayQuiescenceGuard fences each replay.
        let backend = unsafe { B::bind_resources(self.plan, self.capabilities, &external_bindings) }
            .map_err(BindError::Backend)?;
        Ok(BoundExecution {
            plan: self.plan,
            replay_steps: Arc::clone(&self.replay_steps),
            backend,
            _external_bindings: external_bindings,
        })
    }
}

#[inline]
fn prepare_workload_exact_with_capabilities<B: Backend>(
    db: &PlanDb,
    workload: &CanonicalWorkload,
    capabilities: RuntimeCapabilityBinding,
) -> Result<PreparedExecution<B>, BindError<B::Error>> {
    workload.validate().map_err(BindError::InvalidPlan)?;
    let signature = workload.signature();

    if !capabilities.is_valid() {
        return Err(BindError::InvalidPlan(PlanError::InvalidHardwareContract));
    }
    if capabilities.hardware != workload.hardware {
        return Err(BindError::InvalidPlan(PlanError::InvalidHardwareContract));
    }
    if capabilities.memory_topology != workload.memory_topology {
        return Err(BindError::InvalidPlan(PlanError::InvalidMemoryTopology));
    }
    if capabilities.hardware_profile_hash_hi != signature.hardware_hi
        || capabilities.hardware_profile_hash_lo != signature.hardware_lo
    {
        return Err(BindError::InvalidPlan(PlanError::InvalidHardwareContract));
    }
    if capabilities.memory_topology_hash_hi != signature.memory_topology_hi
        || capabilities.memory_topology_hash_lo != signature.memory_topology_lo
    {
        return Err(BindError::InvalidPlan(PlanError::InvalidMemoryTopology));
    }

    let plan = db.resolve_workload_exact(workload).map_err(|error| match error {
        PlanResolveError::NoExactPlan => BindError::NoExactPlan,
        PlanResolveError::AmbiguousExactPlan => BindError::AmbiguousExactPlan,
    })?;
    if !capabilities.matches_plan(plan) {
        return Err(BindError::InvalidPlan(PlanError::InvalidHardwareContract));
    }
    // Fully resolved immutable replay is constructed ONCE at preparation.
    // No schedule scan, Vec allocation, or step lowering occurs during bind or replay.
    // The Vec capacity reservation above is fallible. On stable Rust 1.99,
    // Arc::new itself has no fallible constructor; its small control-block
    // allocation follows the standard allocation-failure policy. This does
    // not occur on the hot replay path and must not be advertised as a fully
    // OOM-recoverable preparation pipeline.
    let replay_steps = build_replay_steps(plan).map_err(|error| match error {
        PlanError::ReplayProgramAllocationFailed => BindError::InsufficientWorkspace,
        other => BindError::InvalidPlan(other),
    })?;
    Ok(PreparedExecution {
        plan,
        capabilities,
        replay_steps: Arc::new(replay_steps),
        _backend_type: core::marker::PhantomData,
    })
}

#[inline]
fn bind_workload_exact_with_capabilities<'a, B: Backend>(
    db: &PlanDb,
    workload: &CanonicalWorkload,
    capabilities: RuntimeCapabilityBinding,
    external_bindings: Vec<StorageBinding<'a>>,
) -> Result<BoundExecution<'a, B::Bound>, BindError<B::Error>> {
    prepare_workload_exact_with_capabilities::<B>(db, workload, capabilities)?
        .bind(external_bindings)
}

// =============================================================================
// Tensor-first ergonomic fixed-binding accessors (no hidden planner)
// =============================================================================

/// Writable binding of *uninitialized* f32 output storage. Unlike an ordinary
/// Tensor, this owner cannot be read until the caller explicitly proves that
/// the selected frozen program initialized the entire output on success.
impl<const R: usize> crate::tensor_light::tensor::UninitTensor<f32, R>
where
    crate::tensor_light::tensor::Rank<R>: crate::tensor_light::tensor::ValidRank,
{
    /// Create a borrowed host-write binding without a preliminary memset or
    /// allocating a staging Vec. The buffer's owner stays borrowed throughout
    /// `BoundExecution` and its mandatory last-use synchronization.
    ///
    /// # Safety
    /// The frozen plan and native backend must never read unwritten elements;
    /// the caller may invoke `assume_init` only after proving every element
    /// initialized on a successful, quiescent execution. On any partial-write
    /// or failed execution, drop this uninitialized owner without reading it.
    /// The physical owner cannot be moved, resized or freed while bound.
    #[inline]
    pub unsafe fn write_binding(&mut self, storage: StorageId) -> StorageBinding<'_> {
        let buffer = BufferHandle::Host(HostBufferHandle {
            address: self.as_mut_ptr() as usize,
            bytes: self.shape().nbytes(),
        });
        // SAFETY: the owner borrow retains allocation lifetime and exclusive
        // access. The safety conditions above govern native initialization.
        unsafe { StorageBinding::from_raw_read_write(storage, buffer, DType::F32, self) }
    }
}

impl<const R: usize> crate::tensor_light::tensor::Tensor<f32, R>
where
    crate::tensor_light::tensor::Rank<R>: crate::tensor_light::tensor::ValidRank,
{
    /// Prepare one externally bound f32 read slot. The returned binding retains
    /// the ordinary Rust shared borrow until the exact frozen graph finishes.
    /// No copy, GPU transfer, allocation, or scheduling is performed here.
    pub fn read_binding(&self, storage: StorageId) -> StorageBinding<'_> {
        StorageBinding::host_slice(storage, self.as_slice())
    }

    // No safe write_binding(&mut Tensor) here: a native partial-write error
    // would leave an ordinary ready Tensor handle falsely readable. Until
    // storage-level poison/epoch tracking and full-overwrite recovery exist,
    // only caller-owned external destination slices can use the explicit
    // EXTERNAL_MAY_PARTIALLY_MODIFY borrowed binding facility. This prevents
    // the Tensor-first facade from claiming a nonexistent poisoning guarantee.
}

#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/physical_alias_tests.rs"]
mod physical_alias_tests;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/replay_preallocation_tests.rs"]
mod replay_preallocation_tests;
#[cfg(test)]
#[path = "../../../tests/tensor_light/unit/execution/runtime/replay_quiescence_guard_tests.rs"]
mod replay_quiescence_guard_tests;
