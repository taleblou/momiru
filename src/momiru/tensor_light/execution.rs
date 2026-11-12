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
// Subcomponent:  Core and Facade
// Module:        momiru::tensor_light::execution
// Source:        src/tensor_light/execution.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Unified UASDEA semantic dataflow, immutable execution plan and deterministic
//   frozen-plan runtime facade.
//
//   This edition uses f32 host tensors with compile-time ranks 0 through 8;
//   numerical graph execution follows frozen UASDEA plans.
//
// Responsibilities:
//   - Connect semantic IR, target descriptors, static plans and runtime replay.
//   - Model CPU, GPU and hybrid placement within one plan abstraction.
//   - Preserve the separation between offline specialization and runtime
//     execution.
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

//! UASDEA immutable static-dataflow records and frozen-plan runtime contracts.
//!
//! A canonical semantic IR is specialized offline into one immutable frozen
//! execution-plan representation. CPU, GPU, and CPU-GPU are placement mappings
//! of that same model. Runtime performs exact binding and deterministic replay.


use core::hash::{Hash, Hasher};

// =============================================================================
// Canonical identity primitives
// =============================================================================

/// Identity schema mixed into every canonical digest. Changing the canonical
/// encoding deliberately invalidates older seals/signatures instead of letting
/// them silently compare equal under a new runtime.
const IDENTITY_SCHEMA_VERSION: u64 = 18;

/// Process/release-local NON-CRYPTOGRAPHIC identity hasher.
///
/// The primitive integer writers below are little-endian, but callers feed
/// this hasher via Rust's `Hash` trait (including derived Hash on nested plan
/// records). Rust explicitly does NOT guarantee that those encodings are
/// portable across compiler versions or target platforms. Thus the resulting
/// digest is NEVER a canonical cross-toolchain wire hash, signature, or an
/// authentication authority. A real signed FrozenExecutionPlan catalog needs
/// an explicitly specified, versioned byte encoder and a cryptographic digest
/// plus signature verification BEFORE public runtime plan loading.
///
/// Identity schema 18 marks the stricter indexed-op admission boundary.
pub(crate) struct StableIdentityHasher {
    state: u64,
}

impl StableIdentityHasher {
    #[inline(always)]
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ IDENTITY_SCHEMA_VERSION.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        }
    }
}

impl Hasher for StableIdentityHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut i = 0usize;
        while i < bytes.len() {
            self.state ^= bytes[i] as u64;
            self.state = self.state.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        self.state ^= self.state >> 32;
        self.state = self.state.wrapping_mul(0xd6e8_feb8_6659_fd93);
        self.state ^= self.state >> 29;
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_u16(&mut self, i: u16) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_u32(&mut self, i: u32) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_u64(&mut self, i: u64) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_u128(&mut self, i: u128) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_usize(&mut self, i: usize) { self.write(&(i as u64).to_le_bytes()); }
    #[inline(always)]
    fn write_i8(&mut self, i: i8) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_i16(&mut self, i: i16) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_i32(&mut self, i: i32) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_i64(&mut self, i: i64) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_i128(&mut self, i: i128) { self.write(&i.to_le_bytes()); }
    #[inline(always)]
    fn write_isize(&mut self, i: isize) { self.write(&(i as i64).to_le_bytes()); }
}

/// Feeds two independent, domain-separated identity streams in ONE traversal
/// of a potentially large canonical graph. Every write is forwarded to the
/// original hashers without changing their bytes, seeds or finalization.
/// In particular, forwarding primitive writers preserves the existing
/// platform-local representation contract and identity schema.
struct PairedIdentityHasher {
    hi: StableIdentityHasher,
    lo: StableIdentityHasher,
}

impl Hasher for PairedIdentityHasher {
    #[inline(always)]
    fn finish(&self) -> u64 { self.hi.finish() }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.hi.write(bytes);
        self.lo.write(bytes);
    }
    #[inline(always)]
    fn write_u8(&mut self, value: u8) {
        self.hi.write_u8(value);
        self.lo.write_u8(value);
    }
    #[inline(always)]
    fn write_u16(&mut self, value: u16) {
        self.hi.write_u16(value);
        self.lo.write_u16(value);
    }
    #[inline(always)]
    fn write_u32(&mut self, value: u32) {
        self.hi.write_u32(value);
        self.lo.write_u32(value);
    }
    #[inline(always)]
    fn write_u64(&mut self, value: u64) {
        self.hi.write_u64(value);
        self.lo.write_u64(value);
    }
    #[inline(always)]
    fn write_u128(&mut self, value: u128) {
        self.hi.write_u128(value);
        self.lo.write_u128(value);
    }
    #[inline(always)]
    fn write_usize(&mut self, value: usize) {
        self.hi.write_usize(value);
        self.lo.write_usize(value);
    }
    #[inline(always)]
    fn write_i8(&mut self, value: i8) {
        self.hi.write_i8(value);
        self.lo.write_i8(value);
    }
    #[inline(always)]
    fn write_i16(&mut self, value: i16) {
        self.hi.write_i16(value);
        self.lo.write_i16(value);
    }
    #[inline(always)]
    fn write_i32(&mut self, value: i32) {
        self.hi.write_i32(value);
        self.lo.write_i32(value);
    }
    #[inline(always)]
    fn write_i64(&mut self, value: i64) {
        self.hi.write_i64(value);
        self.lo.write_i64(value);
    }
    #[inline(always)]
    fn write_i128(&mut self, value: i128) {
        self.hi.write_i128(value);
        self.lo.write_i128(value);
    }
    #[inline(always)]
    fn write_isize(&mut self, value: isize) {
        self.hi.write_isize(value);
        self.lo.write_isize(value);
    }
}

#[inline]
#[doc(hidden)]
pub fn stable_hash_pair<T: Hash + ?Sized>(value: &T, domain: u64) -> (u64, u64) {
    let mut paired = PairedIdentityHasher {
        hi: StableIdentityHasher::new(0xcbf2_9ce4_8422_2325 ^ domain),
        lo: StableIdentityHasher::new(0x8422_2325_cbf2_9ce4 ^ domain.rotate_left(23)),
    };
    value.hash(&mut paired);
    (paired.hi.finish(), paired.lo.finish())
}

#[cfg(test)]
#[path = "../../tests/tensor_light/unit/execution/paired_identity_tests.rs"]
mod paired_identity_tests;
// =============================================================================
// Shared execution identity and validation surface
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanError {
    InvalidTensorAlignment,
    InvalidTensorLayout,
    InvalidTensorLayoutRank,
    InvalidSparsitySpec,
    InvalidQuantizationSpec,
    InvalidAccuracyContract,
    InvalidCanonicalGraph,
    InvalidWorkloadIdentity,
    InvalidCandidateIdentity,
    InvalidMemoryTopology,
    InvalidExecutionTopology,
    InvalidOperationSpec,
    /// Indexed native execution requires a sealed, pre-enqueue index validator
    /// and error dependency that this limited runtime cannot currently replay.
    UnsealedIndexPreflight,
    InvalidFusionDataflow,
    TensorByteCountOverflow,
    DuplicateTensorId,
    /// Two signed/frozen records have the same complete canonical workload identity.
    DuplicateFrozenPlanIdentity,
    DuplicateGpuOrdinal,
    DuplicateArenaId,
    DuplicateMemorySlotId,
    DuplicateStorageId,
    DuplicateExecutionContextId,
    DuplicateStreamId,
    DuplicateEventId,
    DuplicateNodeId,
    DuplicateTensorProducer,
    DuplicateTransferId,
    DuplicateTransferBatchId,
    DuplicateSyncId,
    InvalidResourceIdentity,
    InvalidMemoryGrouping,
    MissingScheduleItem,
    DuplicateScheduleItem,
    MissingArena,
    MissingMemorySlot,
    MissingTensor,
    MissingStorage,
    MissingExecutionContext,
    MissingStream,
    MissingEvent,
    MissingDependencyNode,
    MissingDataDependency,
    StorageTensorMismatch,
    InvalidTargetIoBinding,
    InvalidStorageAccess,
    StorageTooSmall,
    InvalidPhysicalStorage,
    TransferFormatMismatch,
    InvalidMemoryAlignment,
    InvalidArenaAlignment,
    MemorySlotOutOfBounds,
    MemorySlotMisaligned,
    InvalidMemoryLifetime,
    MemoryAliasConflict,
    InvalidArenaPolicy,
    InvalidHardwareContract,
    InvalidLibraryAlgorithm,
    InvalidArtifact,
    InvalidPlanSeal,
    InvalidPlanEvidence,
    /// Host resource failure while reserving the fixed prebound replay program.
    ReplayProgramAllocationFailed,
    /// Initialization-time failure reserving the collision-safe exact-plan key index.
    PlanLookupIndexAllocationFailed,
    InvalidDependencyOrder,
    DuplicateCpuWorker,
    InvalidCpuWorkerPlan,
    InvalidWorkRange,
    InvalidCpuGpuMapping,
    InvalidCpuGpuPartitionAxis,
    MissingCpuMapping,
    MissingGpuMapping,
    CpuMappingOnCpuAbsentHardware,
    GpuMappingOnGpuAbsentHardware,
    CpuFeatureContractMismatch,
    GpuFeatureContractMismatch,
    GpuDeviceMismatch,
    GpuComputeCapabilityMismatch,
    InvalidGpuGrid,
    InvalidGpuBlock,
    InvalidGpuWarpPlan,
    InvalidGpuCluster,
    InvalidGpuStream,
    InvalidGpuExecutionContext,
    InvalidSmPartition,
    InvalidRustCodegen,
    InvalidGpuCodegen,
    InvalidKernelAssumption,
    InvalidKernelFamily,
    InvalidWorkspacePlan,
    InvalidHostBoundary,
    InvalidCpuTuning,
    InvalidGpuTuning,
    InvalidMemoryPool,
    InvalidGpuLaunchAttribute,
    InvalidGpuGraphSegment,
    InvalidL2AccessPolicy,
    InvalidMemorySyncDomain,
    InvalidClusterPolicy,
    InvalidZeroCopyStorage,
    InvalidPinnedCopyStorage,
    InvalidPageableCopyStorage,
    InvalidUnifiedPrefetchStorage,
    InvalidManagedDiscardStorage,
    InvalidTransferTensor,
    InvalidTransferSize,
    InvalidTransferBatch,
    InvalidMemcpyAttributes,
    InvalidSyncReference,
    DuplicateExternalBinding,
    MissingExternalBinding,
    UnexpectedExternalBinding,
    InvalidExternalBinding,
}

impl core::fmt::Display for PlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlanError {}

// Physical/runtime resource identifiers belong to the execution layer, not the
// canonical semantic IR. Keeping them here avoids IR <-> target/plan cycles.
pub type StorageId = u32;
pub type ArenaId = u32;
pub type MemorySlotId = u32;
pub type StreamId = u32;
pub type EventId = u32;
pub type BarrierId = u32;
pub type TransferId = u32;
pub type TransferBatchId = u32;
pub type SyncId = u32;
pub type ExecutionContextId = u32;

/// A slot can require `slot_alignment` only when the allocator's guaranteed
/// *base* alignment is at least that large. Alignment of an offset relative
/// to a weakly aligned arena does not establish physical pointer alignment.
/// Both alignments are power-of-two by the plan's structural validation.
#[inline(always)]
pub(crate) const fn arena_guarantees_slot_alignment(
    arena_alignment: usize,
    slot_alignment: usize,
    offset_bytes: usize,
) -> bool {
    arena_alignment.is_power_of_two()
        && slot_alignment.is_power_of_two()
        && arena_alignment >= slot_alignment
        && offset_bytes % slot_alignment == 0
}

#[cfg(test)]
#[path = "../../tests/tensor_light/unit/execution/slot_alignment_regression.rs"]
mod slot_alignment_regression;
mod ir;
mod plan;
mod runtime;
mod target;
mod validation;
pub mod cpu;

// Candidate search and benchmarking are optional and offline-only (feature `aot`).

pub use ir::*;
pub use plan::*;
pub use runtime::*;
pub use target::*;

