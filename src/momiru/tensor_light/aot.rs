// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT
//
// ==============================================================================
// MOMIRU | OFFLINE AOT
// ==============================================================================
//
// Project:       Momiru
// Component:     Execution
// Subcomponent:  Offline AOT Specialization
// Module:        momiru::tensor_light::aot
// Source:        src/tensor_light/aot.rs
// Language:      Rust
// Architecture:  UASDEA
//
// ------------------------------------------------------------------------------
// OVERVIEW
// ------------------------------------------------------------------------------
//
// Description:
//   Offline-only plan specialization, candidate seeding, cost pruning and
//   numerical selection for UASDEA workloads.
//
// Responsibilities:
//   - Generate and filter candidate execution configurations offline.
//   - Collect benchmark and calibration evidence for candidate comparison.
//   - Reject numerically invalid or unsupported candidates before selection.
//   - Keep offline tuning out of the default runtime replay path.
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

//! Offline-only AOT specialization and candidate seeding.
//!
//! Candidate generation, capability gating, cost pruning, benchmarking, noise
//! calibration, numerical verification, and winner selection terminate here.
//! The default runtime does not compile this optional module.

use core::cmp::Ordering;
use std::collections::HashSet;

use crate::tensor_light::execution::{
    CanonicalWorkload, DenseLinalgKind, Operation, PadMode, ResampleMode, ReusePattern, ScalarElementwiseKind,
    TensorSpec,
};
use crate::tensor_light::execution::{stable_hash_pair, PlanError};
use crate::tensor_light::execution::{
    BenchmarkStats, CandidateEvidence, CandidateId, CostEstimate,
    FrozenExecutionPlan, FrozenPlanParts, KernelFamily, NoiseCalibration,
    NumericalVerification, SelectionEvidence, HeldOutConfirmation,
};
use crate::tensor_light::execution::{gpu_feature, CpuHardwareContract, GpuHardwareContract, LibraryProvider};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenCandidate {
    pub id: CandidateId,
    pub parts: FrozenPlanParts,
}

impl FrozenCandidate {
    pub fn from_parts(parts: FrozenPlanParts) -> Result<Self, PlanError> {
        let id = parts.candidate_id();
        if !id.is_valid() {
            return Err(PlanError::InvalidCandidateIdentity);
        }
        Ok(Self { id, parts })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OfflineSelectionPolicy {
    /// Maximum number of cost-ranked candidates to benchmark. Zero is invalid.
    pub max_cost_survivors: usize,
}

#[derive(Debug)]
pub enum OfflinePipelineError<E> {
    InvalidWorkload(PlanError),
    Specializer(E),
    NoGeneratedCandidates,
    InvalidCandidateId,
    InvalidCandidatePlan(PlanError),
    DuplicateCandidateId,
    NoCapabilityCandidate,
    InvalidBenchmark,
    InvalidNoiseCalibration,
    InconsistentBenchmarkEnvironment,
    InvalidNumericalVerification,
    NoNumericallyVerifiedCandidate,
    HeldOutNotConfirmed,
    CandidateCountOverflow,
    InvalidPolicy,
    /// A candidate/audit/ranking buffer could not reserve its bounded capacity.
    /// This error is returned before attempting to append to the corresponding Vec.
    OfflineWorkspaceAllocationFailed,
    InvalidFrozenPlan(PlanError),
}

/// Offline implementation provider. The opaque seed exists only long enough to
/// materialize an immutable candidate payload. Every later pipeline stage sees
/// that exact payload and its centrally-derived CandidateId.
pub trait OfflineSpecializer {
    type CandidateSeed;
    type Error;

    fn generate_candidates(
        &mut self,
        workload: &CanonicalWorkload,
    ) -> Result<Vec<Self::CandidateSeed>, Self::Error>;

    fn materialize_candidate(
        &mut self,
        workload: &CanonicalWorkload,
        seed: &Self::CandidateSeed,
    ) -> Result<FrozenPlanParts, Self::Error>;

    fn capability_gate(
        &mut self,
        workload: &CanonicalWorkload,
        candidate: &FrozenCandidate,
    ) -> Result<bool, Self::Error>;

    fn estimate_cost(
        &mut self,
        workload: &CanonicalWorkload,
        candidate: &FrozenCandidate,
    ) -> Result<CostEstimate, Self::Error>;

    fn benchmark(
        &mut self,
        workload: &CanonicalWorkload,
        candidate: &FrozenCandidate,
    ) -> Result<BenchmarkStats, Self::Error>;

    fn calibrate_noise(
        &mut self,
        workload: &CanonicalWorkload,
        candidate: &FrozenCandidate,
        benchmark: BenchmarkStats,
    ) -> Result<NoiseCalibration, Self::Error>;

    fn verify_numerically(
        &mut self,
        workload: &CanonicalWorkload,
        candidate: &FrozenCandidate,
    ) -> Result<NumericalVerification, Self::Error>;

    /// Run a distinct randomized, host-observed E2E measurement series after
    /// screening is complete. The data provider must attest independent trials
    /// and compare against a held-out baseline and strongest eligible rival.
    /// No native candidate compiler, measurement provider, or held-out result
    /// is shipped by this offline AOT tuner. The restricted built-in CPU
    /// executor is not an implementation of this measurement interface;
    /// callers MUST NOT fabricate success or performance evidence.
    fn confirm_winner_held_out(
        &mut self,
        workload: &CanonicalWorkload,
        winner: &FrozenCandidate,
        strongest_challenger: Option<&FrozenCandidate>,
    ) -> Result<HeldOutConfirmation, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
enum CandidateAuditStage {
    Generated,
    CapabilityRejected,
    CapabilityAccepted,
    CostPruned,
    Benchmarked,
    VerificationRejected,
    Verified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CandidateAudit {
    id: CandidateId,
    stage: CandidateAuditStage,
    cost: Option<CostEstimate>,
    benchmark: Option<BenchmarkStats>,
    calibration: Option<NoiseCalibration>,
    verification: Option<NumericalVerification>,
}

#[derive(Clone, Copy, Debug)]
struct CostedCandidate {
    candidate: FrozenCandidate,
    audit_index: usize,
    cost: CostEstimate,
}

#[derive(Clone, Copy, Debug)]
struct MeasuredCandidate {
    candidate: FrozenCandidate,
    evidence: CandidateEvidence,
}

/// Fallible, bounded metadata buffer for the offline selection pipeline.
/// This is not a tensor kernel and is never linked into the production runtime.
/// The specializer owns allocations for its own generated seed Vec separately.
#[inline]
fn try_candidate_buffer<T>(bound: usize) -> Result<Vec<T>, ()> {
    let mut values = Vec::new();
    values.try_reserve_exact(bound).map_err(|_| ())?;
    Ok(values)
}

fn candidate_id_order(a: CandidateId, b: CandidateId) -> Ordering {
    a.hi.cmp(&b.hi).then_with(|| a.lo.cmp(&b.lo))
}

fn cost_order(a: &CostedCandidate, b: &CostedCandidate) -> Ordering {
    a.cost
        .score
        .cmp(&b.cost.score)
        .then_with(|| a.cost.workspace_bytes.cmp(&b.cost.workspace_bytes))
        .then_with(|| candidate_id_order(a.candidate.id, b.candidate.id))
}

/// Offline ranking uses a single global uncertainty threshold, not a chain
/// of pairwise "is A better than B?" votes. Interval overlap is NOT transitive:
/// A may overlap B, B may overlap C, while A is distinct from C. The old
/// pairwise tournament could thus pick a different frozen winner when a
/// generator emitted exactly the same candidates in a different order.
///
/// These are screening measurements only; the winner still needs a distinct
/// held-out confirmation before freezing. This private record allows testing
/// the ranking algorithm without fabricating native plans or benchmark proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScreeningRank {
    id: CandidateId,
    median_ns: u64,
    p95_ns: u64,
    uncertainty_ns: u64,
    mad_ns: u64,
    workspace_bytes: usize,
    estimated_shared_memory_bytes: u32,
    estimated_register_bytes: u32,
}

impl ScreeningRank {
    #[inline]
    fn from_measured(item: &MeasuredCandidate) -> Self {
        Self {
            id: item.candidate.id,
            median_ns: item.evidence.benchmark.median_ns,
            p95_ns: item.evidence.benchmark.p95_ns,
            uncertainty_ns: item.evidence.calibration.uncertainty_ns,
            mad_ns: item.evidence.benchmark.mad_ns,
            workspace_bytes: item.evidence.cost.workspace_bytes,
            estimated_shared_memory_bytes: item.evidence.cost.estimated_shared_memory_bytes,
            estimated_register_bytes: item.evidence.cost.estimated_register_bytes,
        }
    }

    /// Select the strongest held-out challenger by the best observed median.
    /// Candidate ID makes ties independent of incoming enumeration order.
    #[inline]
    fn median_key(self) -> (u64, u64, u64, u64) {
        (self.median_ns, self.p95_ns, self.id.hi, self.id.lo)
    }

    /// For measurements not demonstrably dominated by a faster candidate,
    /// prefer the lower-memory, more robust, less costly candidate.
    #[inline]
    fn conservative_key(self) -> (usize, u64, u64, u32, u32, u64, u64) {
        (
            self.workspace_bytes,
            self.uncertainty_ns,
            self.mad_ns,
            self.estimated_shared_memory_bytes,
            self.estimated_register_bytes,
            self.id.hi,
            self.id.lo,
        )
    }

    #[inline]
    fn lower_ns(self) -> u64 {
        self.median_ns.saturating_sub(self.uncertainty_ns)
    }

    #[inline]
    fn upper_ns(self) -> u64 {
        self.median_ns.saturating_add(self.uncertainty_ns)
    }
}

/// Returns (winner, strongest challenger), as indices into `ranks`.
/// `ranks` must be nonempty; the pipeline checks this before calling us.
/// An eligible candidate must NOT be provably slower than *any* other:
/// equivalently its lower uncertainty bound must be <= the minimum of all
/// upper bounds. This global threshold is deterministic, excludes a slower
/// candidate even when its interval overlaps some other noisy candidate, and
/// avoids the non-transitivity of pairwise interval comparisons.
fn select_screening_winner(ranks: &[ScreeningRank]) -> (usize, Option<usize>) {
    let mut best_upper = ranks[0].upper_ns();
    for item in &ranks[1..] {
        best_upper = best_upper.min(item.upper_ns());
    }
    let mut winner: Option<usize> = None;
    for index in 0..ranks.len() {
        if ranks[index].lower_ns() <= best_upper
            && winner.is_none_or(|old| {
                ranks[index].conservative_key() < ranks[old].conservative_key()
            })
        {
            winner = Some(index);
        }
    }
    // At least one lower bound is <= min(upper); the pipeline enforces
    // nonempty, already-validated screening observations.
    let winner = winner.unwrap_or(0);

    // The strongest challenger is the best observed median among *all other*
    // eligible candidates, not the result of another non-transitive pairwise
    // tournament. Its independent held-out comparison is a release gate.
    let mut challenger: Option<usize> = None;
    for index in 0..ranks.len() {
        if index == winner { continue; }
        if challenger.is_none_or(|old| ranks[index].median_key() < ranks[old].median_key()) {
            challenger = Some(index);
        }
    }
    (winner, challenger)
}

pub fn specialize_and_freeze<S: OfflineSpecializer>(
    specializer: &mut S,
    workload: &CanonicalWorkload,
    policy: OfflineSelectionPolicy,
) -> Result<FrozenExecutionPlan, OfflinePipelineError<S::Error>> {
    workload.validate().map_err(OfflinePipelineError::InvalidWorkload)?;
    let signature = workload.signature();
    if !signature.is_complete() || policy.max_cost_survivors == 0 {
        return Err(OfflinePipelineError::InvalidPolicy);
    }

    let seeds = specializer
        .generate_candidates(workload)
        .map_err(OfflinePipelineError::Specializer)?;
    if seeds.is_empty() {
        return Err(OfflinePipelineError::NoGeneratedCandidates);
    }

    let generated_candidates =
        u32::try_from(seeds.len()).map_err(|_| OfflinePipelineError::CandidateCountOverflow)?;
    let mut audits: Vec<CandidateAudit> = try_candidate_buffer(seeds.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;
    let mut materialized: Vec<(FrozenCandidate, usize)> = try_candidate_buffer(seeds.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;
    let mut unique_ids = HashSet::<CandidateId>::new();
    unique_ids.try_reserve(seeds.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;

    let mut s = 0usize;
    while s < seeds.len() {
        let parts = specializer
            .materialize_candidate(workload, &seeds[s])
            .map_err(OfflinePipelineError::Specializer)?;
        let candidate = FrozenCandidate::from_parts(parts)
            .map_err(|_| OfflinePipelineError::InvalidCandidateId)?;
        if !unique_ids.insert(candidate.id) {
            return Err(OfflinePipelineError::DuplicateCandidateId);
        }
        audits.push(CandidateAudit {
            id: candidate.id,
            stage: CandidateAuditStage::Generated,
            cost: None,
            benchmark: None,
            calibration: None,
            verification: None,
        });
        materialized.push((candidate, audits.len() - 1));
        s += 1;
    }

    let mut costed: Vec<CostedCandidate> = try_candidate_buffer(seeds.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;
    let mut capability_survivors = 0u32;
    let mut m = 0usize;
    while m < materialized.len() {
        let (candidate, audit_index) = materialized[m];

        // Structural/semantic validity is independent of target acceptance.
        // Preflight the exact materialized payload before capability gating so
        // a specializer never gets to inspect/accept a malformed candidate.
        candidate
            .parts
            .validate_for(*workload)
            .map_err(OfflinePipelineError::InvalidCandidatePlan)?;

        let accepted = specializer
            .capability_gate(workload, &candidate)
            .map_err(OfflinePipelineError::Specializer)?;
        if !accepted {
            audits[audit_index].stage = CandidateAuditStage::CapabilityRejected;
            m += 1;
            continue;
        }
        capability_survivors = capability_survivors
            .checked_add(1)
            .ok_or(OfflinePipelineError::CandidateCountOverflow)?;
        audits[audit_index].stage = CandidateAuditStage::CapabilityAccepted;
        let cost = specializer
            .estimate_cost(workload, &candidate)
            .map_err(OfflinePipelineError::Specializer)?;
        audits[audit_index].cost = Some(cost);
        costed.push(CostedCandidate { candidate, audit_index, cost });
        m += 1;
    }
    if costed.is_empty() {
        return Err(OfflinePipelineError::NoCapabilityCandidate);
    }

    // Cost/space/unique-id keys define a TOTAL ordering, so stable sorting
    // does not add useful semantics. In-place unstable sort avoids scratch
    // allocation and is deterministic because candidate IDs are unique.
    costed.sort_unstable_by(cost_order);
    if costed.len() > policy.max_cost_survivors {
        let mut p = policy.max_cost_survivors;
        while p < costed.len() {
            audits[costed[p].audit_index].stage = CandidateAuditStage::CostPruned;
            p += 1;
        }
        costed.truncate(policy.max_cost_survivors);
    }
    let cost_model_survivors =
        u32::try_from(costed.len()).map_err(|_| OfflinePipelineError::CandidateCountOverflow)?;

    let mut measured: Vec<MeasuredCandidate> = try_candidate_buffer(costed.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;
    let mut benchmarked_candidates = 0u32;
    let mut noise_calibrated_candidates = 0u32;
    let mut numerically_verified_candidates = 0u32;
    let mut verification_passed_candidates = 0u32;
    let mut calibration_environment: Option<(u64, u64)> = None;

    let mut c = 0usize;
    while c < costed.len() {
        let entry = costed[c];

        // Numerics FIRST: a malformed, incompatible, or numerically invalid
        // candidate cannot enter the benchmark phase merely because its cost
        // model looks promising. This is the normative AOT v24 ordering.
        let verification = specializer
            .verify_numerically(workload, &entry.candidate)
            .map_err(OfflinePipelineError::Specializer)?;
        if !verification.is_valid() {
            return Err(OfflinePipelineError::InvalidNumericalVerification);
        }
        numerically_verified_candidates = numerically_verified_candidates
            .checked_add(1)
            .ok_or(OfflinePipelineError::CandidateCountOverflow)?;
        audits[entry.audit_index].verification = Some(verification);
        if !verification.satisfies(workload) {
            audits[entry.audit_index].stage = CandidateAuditStage::VerificationRejected;
            c += 1;
            continue;
        }
        verification_passed_candidates = verification_passed_candidates
            .checked_add(1)
            .ok_or(OfflinePipelineError::CandidateCountOverflow)?;

        let benchmark = specializer
            .benchmark(workload, &entry.candidate)
            .map_err(OfflinePipelineError::Specializer)?;
        if !benchmark.is_valid() {
            return Err(OfflinePipelineError::InvalidBenchmark);
        }
        benchmarked_candidates = benchmarked_candidates
            .checked_add(1)
            .ok_or(OfflinePipelineError::CandidateCountOverflow)?;
        audits[entry.audit_index].benchmark = Some(benchmark);
        audits[entry.audit_index].stage = CandidateAuditStage::Benchmarked;

        let calibration = specializer
            .calibrate_noise(workload, &entry.candidate, benchmark)
            .map_err(OfflinePipelineError::Specializer)?;
        if !calibration.is_valid() {
            return Err(OfflinePipelineError::InvalidNoiseCalibration);
        }
        let environment = (
            calibration.environment_signature_hi,
            calibration.environment_signature_lo,
        );
        if let Some(expected) = calibration_environment {
            if expected != environment {
                return Err(OfflinePipelineError::InconsistentBenchmarkEnvironment);
            }
        } else {
            calibration_environment = Some(environment);
        }
        noise_calibrated_candidates = noise_calibrated_candidates
            .checked_add(1)
            .ok_or(OfflinePipelineError::CandidateCountOverflow)?;
        audits[entry.audit_index].calibration = Some(calibration);
        audits[entry.audit_index].stage = CandidateAuditStage::Verified;
        measured.push(MeasuredCandidate {
            candidate: entry.candidate,
            evidence: CandidateEvidence {
                id: entry.candidate.id,
                capability_accepted: true,
                cost: entry.cost,
                benchmark,
                calibration,
                verification,
            },
        });
        c += 1;
    }

    if measured.is_empty() {
        return Err(OfflinePipelineError::NoNumericallyVerifiedCandidate);
    }

    // Stable global non-dominated selection using the minimum screened upper
    // bound. Pairwise interval comparisons are non-transitive (v34 regression).
    let mut rank_records: Vec<ScreeningRank> = try_candidate_buffer(measured.len())
        .map_err(|_| OfflinePipelineError::OfflineWorkspaceAllocationFailed)?;
    for item in &measured {
        rank_records.push(ScreeningRank::from_measured(item));
    }
    let (winner_index, challenger_index) = select_screening_winner(&rank_records);
    let winner = measured[winner_index];
    let challenger_id = challenger_index.map(|index| measured[index].candidate.id);
    let challenger = challenger_index.map(|index| &measured[index].candidate);
    let held_out = specializer
        .confirm_winner_held_out(workload, &winner.candidate, challenger)
        .map_err(OfflinePipelineError::Specializer)?;
    if !held_out.is_valid_for(winner.evidence.calibration, winner.candidate.id, challenger_id) {
        return Err(OfflinePipelineError::HeldOutNotConfirmed);
    }
    // Candidate generation order is not semantic. Hash the complete audit set
    // in content-derived candidate-id order so equivalent offline generators
    // produce the same immutable evidence identity.
    // Candidate IDs are verified unique, so a no-allocation total-order sort
    // preserves exact canonical audit ordering across generator permutations.
    audits.sort_unstable_by(|a, b| candidate_id_order(a.id, b.id));
    let candidate_set_hash = stable_hash_pair(&audits.as_slice(), 0x4341_4e44_4155_4449); // CANDAUDI
    let evidence = SelectionEvidence::new(
        signature,
        winner.evidence,
        generated_candidates,
        capability_survivors,
        cost_model_survivors,
        benchmarked_candidates,
        noise_calibrated_candidates,
        numerically_verified_candidates,
        verification_passed_candidates,
        candidate_set_hash,
        held_out,
    );

    FrozenExecutionPlan::freeze_offline(winner.candidate.parts, *workload, evidence)
        .map_err(OfflinePipelineError::InvalidFrozenPlan)
}


// =============================================================================
// Candidate seed policies
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuOfflineSeedPolicy {
    pub rank_specialization_limit: u8,
    pub small_tensor_max_elements: u64,
    pub parallel_min_elements: u64,
    pub parallel_tasks_per_worker: u8,
    pub min_parallel_chunk_elements: u64,
    pub require_out_variant_for_hot_path: bool,
    pub seed_persistent_weight_prepack: bool,
    /// Include exact CPU vendor-library algorithms as offline candidates.
    pub seed_vendor_libraries: bool,
}

impl Default for CpuOfflineSeedPolicy {
    fn default() -> Self {
        Self {
            rank_specialization_limit: 5,
            small_tensor_max_elements: 4_096,
            parallel_min_elements: 1_048_576,
            parallel_tasks_per_worker: 4,
            min_parallel_chunk_elements: 32_768,
            require_out_variant_for_hot_path: true,
            seed_persistent_weight_prepack: true,
            seed_vendor_libraries: true,
        }
    }
}

const CPU_VENDOR_NONE: &[LibraryProvider] = &[];
const CPU_VENDOR_MATMUL: &[LibraryProvider] = &[
    LibraryProvider::Mkl,
    LibraryProvider::OpenBlas,
    LibraryProvider::Blis,
    LibraryProvider::Accelerate,
    LibraryProvider::Blas,
    LibraryProvider::OneDnn,
];
const CPU_VENDOR_LINALG: &[LibraryProvider] = &[
    LibraryProvider::Mkl,
    LibraryProvider::OpenBlas,
    LibraryProvider::Accelerate,
    LibraryProvider::Blas,
];
const CPU_VENDOR_FFT: &[LibraryProvider] = &[
    LibraryProvider::Mkl,
    LibraryProvider::Accelerate,
];
const CPU_VENDOR_DNN: &[LibraryProvider] = &[LibraryProvider::OneDnn];

/// Exact CPU library providers worth materializing for one semantic operation.
/// The returned list is a search-space seed only; benchmark and numerical
/// verification still decide whether any provider survives into the frozen plan.
pub const fn cpu_vendor_provider_seeds(
    operation: Operation,
    policy: CpuOfflineSeedPolicy,
) -> &'static [LibraryProvider] {
    if !policy.seed_vendor_libraries {
        return CPU_VENDOR_NONE;
    }
    match operation {
        Operation::MatMul { .. } | Operation::TensorContract { .. } => CPU_VENDOR_MATMUL,
        Operation::DenseLinearAlgebra(_) => CPU_VENDOR_LINALG,
        Operation::FourierTransform(_) => CPU_VENDOR_FFT,
        Operation::Convolution(_)
        | Operation::Attention(_)
        | Operation::Normalization(_)
        | Operation::Pooling(_)
        | Operation::Softmax { .. } => CPU_VENDOR_DNN,
        _ => CPU_VENDOR_NONE,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuOfflineSeedPolicy {
    pub default_threads_per_block: u16,
    pub max_elementwise_blocks: u32,
    pub gemm_large_tile: u16,
    /// Enumerate Hopper+ one-dimensional cp.async.bulk candidates when capability-gated.
    pub seed_bulk_async_copy: bool,
    /// Enumerate tiled/im2col TMA candidates for compatible static convolution/data-movement nodes.
    pub seed_tma_tensor_maps: bool,
    /// Consider device constant memory for small persistent read-only broadcast operands.
    pub seed_constant_memory: bool,
    /// Include exact vendor-library algorithms (cuBLAS/cuBLASLt/cuFFT/cuSOLVER/cuDNN/...).
    pub seed_vendor_libraries: bool,
    /// Allow offline CompileIQ/advanced-controls variants to compete as distinct AOT artifacts.
    pub seed_compileiq_codegen: bool,
    /// Search-space guard only; the exact hardware limit still capability-gates the materialized plan.
    pub constant_memory_seed_max_bytes: u32,
}

impl Default for GpuOfflineSeedPolicy {
    fn default() -> Self {
        Self {
            default_threads_per_block: 256,
            max_elementwise_blocks: 4_096,
            gemm_large_tile: 32,
            seed_bulk_async_copy: true,
            seed_tma_tensor_maps: true,
            seed_constant_memory: true,
            seed_vendor_libraries: true,
            seed_compileiq_codegen: true,
            constant_memory_seed_max_bytes: 64 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuAdvancedCandidateSeeds {
    /// Hopper+ one-dimensional bulk asynchronous copies should be enumerated.
    pub bulk_async_copy: bool,
    /// Tiled TMA descriptors should be enumerated for compatible static accesses.
    pub tma_tiled: bool,
    /// Convolution may enumerate TMA im2col descriptors.
    pub tma_im2col: bool,
    /// Blackwell+ convolution may enumerate wide-im2col descriptors.
    pub tma_im2col_wide: bool,
    /// Small persistent read-only operands may be promoted to constant memory.
    pub constant_memory: bool,
    /// Offline compiler-control variants may be emitted as distinct native artifacts.
    pub compileiq_codegen: bool,
    /// Exact vendor providers worth materializing and benchmarking for this semantic op.
    pub vendor_providers: &'static [LibraryProvider],
}

const GPU_VENDOR_NONE: &[LibraryProvider] = &[];
const GPU_VENDOR_MATMUL: &[LibraryProvider] = &[
    LibraryProvider::CublasLt,
    LibraryProvider::Cublas,
];
const GPU_VENDOR_FFT: &[LibraryProvider] = &[LibraryProvider::CuFft];
const GPU_VENDOR_LINALG: &[LibraryProvider] = &[LibraryProvider::CuSolver];
const GPU_VENDOR_CONV: &[LibraryProvider] = &[LibraryProvider::Cudnn];
const GPU_VENDOR_TENSOR_CONTRACT: &[LibraryProvider] = &[
    LibraryProvider::CuTensor,
    LibraryProvider::CublasLt,
];
const GPU_VENDOR_ATTENTION: &[LibraryProvider] = &[LibraryProvider::Cudnn];
const GPU_VENDOR_COLLECTIVE: &[LibraryProvider] = &[LibraryProvider::Nccl];
const GPU_VENDOR_DNN: &[LibraryProvider] = &[LibraryProvider::Cudnn];

/// Seed the advanced GPU search surface from the exact semantic operation and
/// hardware contract. This function does not select a winner: every `true` bit
/// or provider is only permission for the offline specializer to materialize
/// concrete alternatives that still pass capability gating, cost pruning,
/// benchmarking, noise calibration and numerical verification.
/// Only a native CUDA `sm_*` target is eligible for CUDA-only tools,
/// libraries and architectural mechanisms. HIP `gfx*` devices may have a
/// 32-thread or 64-thread wavefront, but they are never CUDA targets.
/// This is an OFFLINE candidate gate; final native artifact, ABI and feature
/// compatibility must still be proven before freezing any execution plan.
#[inline]
fn is_cuda_candidate_target(hw: &GpuHardwareContract) -> bool {
    if hw.warp_size != 32 {
        return false;
    }
    let Some(arch) = hw.native_arch.strip_prefix("sm_") else {
        return false;
    };
    // `sm_89`, `sm_90a`, `sm_100a` encode compute capability 8.9,
    // 9.0 and 10.0 respectively. An untrusted/mistagged profile must not
    // enable Hopper CUDA instructions on an SM89 native target just because
    // compute_major was incorrectly set to 9. AOT target verification will
    // still perform the complete compiler/device/native-ABI compatibility test.
    let digit_count = arch.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count < 2 {
        return false;
    }
    let (digits, suffix) = arch.split_at(digit_count);
    if !suffix.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return false;
    }
    let Ok(capability) = digits.parse::<u16>() else {
        return false;
    };
    capability / 10 == u16::from(hw.compute_major)
        && capability % 10 == u16::from(hw.compute_minor)
}

pub fn gpu_advanced_candidate_seeds(
    operation: Operation,
    hw: &GpuHardwareContract,
    persistent_read_only_bytes: usize,
    policy: GpuOfflineSeedPolicy,
) -> GpuAdvancedCandidateSeeds {
    let cuda_target = is_cuda_candidate_target(hw);
    let hopper_or_newer = cuda_target && hw.compute_major >= 9;
    let has_tma = hopper_or_newer && hw.features.contains(gpu_feature::TMA);
    // The current vendor list contains NVIDIA CUDA providers only. AMD HIP
    // native/vendor candidates require a separately audited ABI and provider
    // list, not a misleading cuBLAS/cuDNN entry on a gfx target.
    let vendor_providers = if !cuda_target || !policy.seed_vendor_libraries {
        GPU_VENDOR_NONE
    } else {
        match operation {
            Operation::MatMul { .. } => GPU_VENDOR_MATMUL,
            Operation::FourierTransform(_) => GPU_VENDOR_FFT,
            Operation::DenseLinearAlgebra(_) => GPU_VENDOR_LINALG,
            Operation::Convolution(_) => GPU_VENDOR_CONV,
            Operation::TensorContract { .. } => GPU_VENDOR_TENSOR_CONTRACT,
            Operation::Attention(_) => GPU_VENDOR_ATTENTION,
            Operation::Collective(_) => GPU_VENDOR_COLLECTIVE,
            Operation::Normalization(_) | Operation::Pooling(_) => GPU_VENDOR_DNN,
            _ => GPU_VENDOR_NONE,
        }
    };

    let constant_limit = usize::min(
        hw.constant_memory_bytes,
        policy.constant_memory_seed_max_bytes as usize,
    );

    GpuAdvancedCandidateSeeds {
        bulk_async_copy: policy.seed_bulk_async_copy
            && hopper_or_newer
            && hw.features.contains(gpu_feature::BULK_ASYNC_COPY),
        tma_tiled: policy.seed_tma_tensor_maps && has_tma,
        tma_im2col: policy.seed_tma_tensor_maps
            && has_tma
            && hw.features.contains(gpu_feature::TENSOR_MAP_IM2COL)
            && matches!(operation, Operation::Convolution(_)),
        tma_im2col_wide: policy.seed_tma_tensor_maps
            && has_tma
            && hw.compute_major >= 10
            && hw.features.contains(gpu_feature::TENSOR_MAP_IM2COL_WIDE)
            && matches!(operation, Operation::Convolution(_)),
        constant_memory: cuda_target
            && policy.seed_constant_memory
            && persistent_read_only_bytes > 0
            && persistent_read_only_bytes <= constant_limit,
        compileiq_codegen: cuda_target && policy.seed_compileiq_codegen,
        vendor_providers,
    }
}

/// Families suggested by the source SGEMM implementation. A seed is not a
/// selector: candidate generation should still expand nearby alternatives and
/// let offline benchmarking decide the winner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuSgemmSeedFamily {
    FixedTiny,
    BatchedSmall,
    Direct,
    DirectAxpy,
    DirectAxpyAlpha1Beta0,
    Packed4x8,
    Packed6xSimd,
    PrepackedB4x8,
}

impl CpuSgemmSeedFamily {
    #[inline(always)]
    pub const fn kernel_family(self) -> KernelFamily {
        match self {
            Self::FixedTiny => KernelFamily::CpuSgemmFixedTiny,
            Self::BatchedSmall => KernelFamily::CpuSgemmBatchedSmall,
            Self::Direct => KernelFamily::CpuSgemmDirect,
            Self::DirectAxpy | Self::DirectAxpyAlpha1Beta0 => KernelFamily::CpuSgemmDirectAxpy,
            Self::Packed4x8 => KernelFamily::CpuSgemmPacked4x8,
            Self::Packed6xSimd => KernelFamily::CpuSgemmPacked6xSimd,
            Self::PrepackedB4x8 => KernelFamily::CpuSgemmPrepackedB4x8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuSgemmSeed {
    pub family: CpuSgemmSeedFamily,
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub micro_m: u16,
    pub micro_n: u16,
    pub pack_a: bool,
    pub pack_b: bool,
    /// True only for a persistent fixed-B candidate. Candidate materialization
    /// maps this to CpuKernelPlan::persistent_packed_weights.
    pub persistent_packed_weights: bool,
    pub seed_parallel_2d: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuElementwiseCostClass {
    Cheap,
    Expensive,
}

/// Static search-space seeds. Hardware limits must still gate these values.
pub const GPU_THREADS_PER_BLOCK_SEEDS: &[u16] = &[32, 64, 128, 256, 512, 1024];
/// Useful rowwise/scan widths observed in the supplied GPU kernels. These are
/// enumeration seeds only; capability gating and offline benchmark selection
/// still decide the frozen winner.
pub const GPU_ROWWISE_WIDTH_SEEDS: &[u16] = &[32, 64, 128, 256, 512, 1024];
pub const CPU_SGEMM_K_SEEDS: &[u16] = &[32, 64, 128, 192, 256];
pub const CPU_SGEMM_M_SEEDS: &[u16] = &[16, 32, 64, 96, 128];
pub const CPU_SGEMM_N_SEEDS: &[u16] = &[32, 64, 128, 192, 256];

/// Primary family seed for ordinary elementwise work. Broadcast and contiguous
/// paths are kept distinct because the supplied CPU/GPU implementations show
/// materially different address-generation costs. This never dispatches at runtime.
#[inline(always)]
pub const fn elementwise_kernel_family_seed(broadcast_signature: u64) -> KernelFamily {
    if broadcast_signature == 0 {
        KernelFamily::ContiguousElementwise
    } else {
        KernelFamily::BroadcastElementwise
    }
}

/// Logical mask operations have a distinct byte-wide memory/compute profile.
#[inline(always)]
pub const fn mask_elementwise_kernel_family_seed() -> KernelFamily {
    KernelFamily::MaskElementwise
}

/// PReLU with a rank-1 alpha mapped onto the final axis mirrors the dedicated
/// last-dimension kernels in the supplied CPU/GPU implementations.
#[inline(always)]
pub const fn prelu_kernel_family_seed(channelwise_lastdim: bool, broadcast_signature: u64) -> KernelFamily {
    if channelwise_lastdim {
        KernelFamily::ChannelwisePRelu
    } else {
        elementwise_kernel_family_seed(broadcast_signature)
    }
}

#[inline(always)]
pub const fn scalar_elementwise_kernel_family_seed(
    kind: ScalarElementwiseKind,
    scalar0_f32_bits: u32,
) -> KernelFamily {
    match kind {
        ScalarElementwiseKind::Power if scalar0_f32_bits == 0x3fc0_0000 => {
            KernelFamily::ScalarPowerThreeHalves
        }
        _ => KernelFamily::ScalarElementwise,
    }
}

#[inline(always)]
pub const fn select_kernel_family_seed() -> KernelFamily {
    KernelFamily::Select
}

/// View-producing transforms should always enumerate an alias candidate when
/// the storage planner can prove the exact signed/zero-stride mapping. The
/// materialized family remains in the offline candidate set because a copy can
/// improve downstream locality enough to win end-to-end.
#[inline(always)]
pub const fn view_kernel_family_seed(can_alias: bool, materialized: KernelFamily) -> KernelFamily {
    if can_alias { KernelFamily::AliasView } else { materialized }
}

#[inline(always)]
pub const fn expand_kernel_family_seed(can_alias: bool) -> KernelFamily {
    view_kernel_family_seed(can_alias, KernelFamily::ExpandMaterialize)
}

#[inline(always)]
pub const fn flip_kernel_family_seed(
    can_alias: bool,
    rank: usize,
    contiguous: bool,
    flipped_axes: usize,
) -> KernelFamily {
    if can_alias {
        KernelFamily::AliasView
    } else if rank == 2 && contiguous && flipped_axes == 1 {
        KernelFamily::FlipContiguous2d
    } else {
        KernelFamily::FlipGeneric
    }
}

#[inline(always)]
pub const fn concat_kernel_family_seed(input_count: usize) -> KernelFamily {
    if input_count == 2 {
        KernelFamily::Concat2
    } else if input_count == 4 {
        KernelFamily::Concat4
    } else {
        KernelFamily::ConcatGeneric
    }
}

#[inline(always)]
pub const fn stack_kernel_family_seed(input_count: usize) -> KernelFamily {
    if input_count == 2 {
        KernelFamily::Stack2
    } else if input_count == 4 {
        KernelFamily::Stack4
    } else {
        KernelFamily::StackGeneric
    }
}

#[inline(always)]
pub const fn pad_kernel_family_seed(mode: PadMode, contiguous: bool) -> KernelFamily {
    if matches!(mode, PadMode::Constant) && contiguous {
        KernelFamily::PadConstantContiguous
    } else {
        KernelFamily::PadMapped
    }
}

#[inline(always)]
pub const fn resample_kernel_family_seed(mode: ResampleMode, scale: u32) -> KernelFamily {
    match mode {
        ResampleMode::Nearest => KernelFamily::ResampleNearest,
        ResampleMode::Bilinear => KernelFamily::ResampleBilinear,
        ResampleMode::MeanDownsample if scale == 2 => KernelFamily::DownsampleMeanScale2,
        ResampleMode::MeanDownsample => KernelFamily::DownsampleMeanGeneric,
    }
}

#[inline(always)]
pub const fn sliding_window_kernel_family_seed(can_alias: bool) -> KernelFamily {
    view_kernel_family_seed(can_alias, KernelFamily::SlidingWindowContiguous)
}

/// Both supplied CPU and GPU implementations share this exact rank-1 semantic.
/// The factor is static in the canonical operation, so generated artifacts can
/// strength-reduce modulo/division or fully unroll small factors offline.
#[inline(always)]
pub const fn zero_insert_1d_kernel_family_seed() -> KernelFamily {
    KernelFamily::ZeroInsert1d
}

/// Supplied CPU/GPU code has a dedicated path when all repeated dimensions are
/// singleton and only the final axis expands; otherwise AOT emits periodic
/// modulo indexing with shape/repeat constants embedded in the artifact.
#[inline(always)]
pub const fn tile_kernel_family_seed(singleton_last_only: bool) -> KernelFamily {
    if singleton_last_only {
        KernelFamily::RepeatSingletonLast
    } else {
        KernelFamily::TileMaterialize
    }
}

#[inline(always)]
pub const fn repeat_interleave_kernel_family_seed(contiguous_last_axis: bool) -> KernelFamily {
    if contiguous_last_axis {
        KernelFamily::RepeatInterleaveLast
    } else {
        KernelFamily::RepeatInterleaveGeneric
    }
}

/// The CPU implementation includes an integral-image path for rank-2 mean
/// downsampling. It is deliberately a candidate, not a runtime heuristic: the
/// prefix-sum setup only wins for some spatial sizes/scales.
#[inline(always)]
pub const fn downsample_mean_integral2d_candidate_seed(rank: usize, scale: u32) -> bool {
    rank == 2 && scale > 1
}

/// Primary reduction seed. Small one-output row reductions favor a direct
/// rowwise tree; very large reductions receive a two-pass/workspace candidate.
/// Neighboring families should still be generated and benchmarked offline.
#[inline(always)]
pub const fn reduction_kernel_family_seed(reduced_extent: u64, output_elements: u64) -> KernelFamily {
    if output_elements > 1 && reduced_extent <= 4_096 {
        KernelFamily::RowwiseReduction
    } else if reduced_extent >= 131_072 {
        KernelFamily::TwoPassReduction
    } else {
        KernelFamily::TreeReduction
    }
}

#[inline(always)]
pub const fn normalization_kernel_family_seed() -> KernelFamily {
    KernelFamily::FusedNormalization
}

#[inline(always)]
pub const fn softmax_kernel_family_seed(log: bool) -> KernelFamily {
    if log {
        KernelFamily::FusedLogSoftmax
    } else {
        KernelFamily::FusedSoftmax
    }
}

/// Prefix kernels in the supplied GPU implementation use a one-block fast path
/// for short contiguous rows and tiled/segmented work for larger/general axes.
#[inline(always)]
pub const fn scan_kernel_family_seed(axis_is_last_contiguous: bool, reduced_extent: u64) -> KernelFamily {
    if axis_is_last_contiguous && reduced_extent <= 1_024 {
        KernelFamily::ParallelPrefixScan
    } else {
        KernelFamily::SegmentedScan
    }
}

/// The supplied GPU sort uses shared-memory/bitonic-style candidates for bounded
/// row widths while CPU/general-axis paths use comparison sorting. Both remain
/// offline candidates; this function is only the initial enumeration hint.
#[inline(always)]
pub const fn sort_kernel_family_seed(axis_is_last_contiguous: bool, axis_extent: u64) -> KernelFamily {
    if axis_is_last_contiguous && axis_extent <= 1_024 {
        KernelFamily::BitonicSort
    } else {
        KernelFamily::ComparisonSort
    }
}

/// The supplied GPU TopK uses direct bounded-k selection through k=32 and a
/// full-sort/trim fallback above that point. These are only offline seeds; the
/// benchmarked target-specific winner remains authoritative.
#[inline(always)]
pub const fn topk_kernel_family_seed(k: u32) -> KernelFamily {
    if k <= 32 {
        KernelFamily::TopKSelection
    } else {
        KernelFamily::TopKSortTrim
    }
}

/// Only emit CPU worker candidates for an actually enumerated, internally
/// consistent processor topology. Zero represents an invalid/unavailable CPU
/// candidate, never an invented one-worker fallback. The full hardware
/// contract, per-worker eligible ISA set, affinity and NUMA policy must still
/// be verified before offline freeze; this seed does not prove those facts.
#[inline(always)]
pub fn cpu_primary_worker_seed(hw: &CpuHardwareContract, prefer_physical_cores: bool) -> u16 {
    if !hw.has_consistent_processor_inventory() {
        return 0;
    }
    if prefer_physical_cores {
        hw.physical_core_count
    } else {
        hw.logical_cpu_count
    }
}

/// Fixed offline CPU task decomposition. Unlike the previous hard-coded
/// four tasks per worker, this respects the published candidate policy. A
/// zero worker count or zero tasks/worker disables this candidate family;
/// it never manufactures a worker or chooses a different algorithm online.
#[inline(always)]
pub fn cpu_parallel_task_seed_with_policy(
    n: u64,
    workers: u16,
    min_chunk: u64,
    policy: CpuOfflineSeedPolicy,
) -> u32 {
    if n == 0 || workers == 0 || policy.parallel_tasks_per_worker == 0 {
        return 0;
    }
    let mut tasks = u64::from(workers) * u64::from(policy.parallel_tasks_per_worker);
    if min_chunk > 0 {
        let by_chunk = n.div_ceil(min_chunk);
        tasks = tasks.min(by_chunk);
    }
    tasks = tasks.min(n);
    // Upper bound is u16::MAX * u8::MAX < u32::MAX.
    u32::try_from(tasks).unwrap_or(u32::MAX)
}

/// Compatibility seed with the default offline policy (4 tasks/worker).
/// Call `cpu_parallel_task_seed_with_policy` to enumerate other static
/// scheduling candidates without runtime performance tuning.
#[inline(always)]
pub fn cpu_parallel_task_seed(n: u64, workers: u16, min_chunk: u64) -> u32 {
    cpu_parallel_task_seed_with_policy(n, workers, min_chunk, CpuOfflineSeedPolicy::default())
}

#[inline(always)]
pub fn cpu_parallel_grain_seed(op_cost: u32) -> u64 {
    if op_cost <= 1 {
        return 32_768;
    }
    (32_768u64 / u64::from(op_cost)).max(4_096)
}

#[inline(always)]
pub fn cpu_rank_should_receive_specialized_candidates(rank: usize, policy: CpuOfflineSeedPolicy) -> bool {
    rank > 0
        && rank <= crate::tensor_light::MAX_RANK
        && rank <= usize::from(policy.rank_specialization_limit)
}

#[inline(always)]
pub fn cpu_is_small_tensor_seed(n: u64, policy: CpuOfflineSeedPolicy) -> bool {
    n <= policy.small_tensor_max_elements
}

#[inline(always)]
pub fn cpu_should_seed_parallel(n: u64, workers: u16, policy: CpuOfflineSeedPolicy) -> bool {
    n > 0
        && workers > 1
        && policy.parallel_tasks_per_worker > 0
        && n >= policy.parallel_min_elements
}

#[inline(always)]
pub fn cpu_chunk_elements_from_kib_seed(n: u64, kib_hint: u64) -> u64 {
    if n == 0 {
        return 1;
    }
    let bytes = kib_hint.saturating_mul(1024);
    let mut chunk = (bytes / 8).clamp(4_096, 2_000_000);
    chunk = chunk.min(n).max(1);
    chunk
}

/// Primary copy-family seed from canonical layout. Candidate generation should
/// still include alternatives when the cost model considers them plausible.
#[inline(always)]
pub fn copy_kernel_family_seed(input: &TensorSpec) -> KernelFamily {
    if input.layout.is_row_major_contiguous_for(&input.shape) {
        KernelFamily::ContiguousCopy
    } else {
        KernelFamily::StridedCopy
    }
}

/// Primary gather-family seed mirroring useful layout-specialized paths from
/// the source implementation. It is an enumeration hint, not runtime dispatch.
#[inline(always)]
pub fn gather_kernel_family_seed(input: &TensorSpec) -> KernelFamily {
    if input.shape.rank() == 4 && input.layout.is_channels_last_contiguous_for(&input.shape) {
        KernelFamily::ChannelsLastGather4d
    } else if input.shape.rank() == 5 && input.layout.is_channels_last_contiguous_for(&input.shape) {
        KernelFamily::ChannelsLastGather5d
    } else if input.layout.is_row_major_contiguous_for(&input.shape) {
        KernelFamily::ContiguousGather
    } else {
        KernelFamily::FancyGatherOneAxis
    }
}

#[inline(always)]
pub fn cpu_sgemm_primary_seed(
    m: u64,
    k: u64,
    n: u64,
    workers: u16,
    b_row_contiguous: bool,
    c_row_contiguous: bool,
    alpha_is_one: bool,
    beta_is_zero: bool,
) -> CpuSgemmSeed {
    let work = m.saturating_mul(n).saturating_mul(k);

    // Exact generated hot-shape kernels in the supplied source. Because the
    // workload shape is static, this remains an AOT candidate rather than a
    // replay-time branch.
    if (m == 2 && k == 2 && n == 2) || (m == 3 && k == 3 && n == 3) {
        return CpuSgemmSeed {
            family: CpuSgemmSeedFamily::FixedTiny,
            tile_m: m as u16,
            tile_n: n as u16,
            tile_k: k as u16,
            micro_m: m as u16,
            micro_n: n as u16,
            pack_a: false,
            pack_b: false,
            persistent_packed_weights: false,
            seed_parallel_2d: false,
        };
    }

    // Mirrors the newer supplied SGEMM source more faithfully. In particular,
    // M/N in 5..=8 are not forced down the direct path for very large wide/tall
    // matrices, where packing can amortize and win. These are only enumeration
    // seeds; offline benchmarking is still authoritative.
    let direct = m > 0
        && n > 0
        && k > 0
        && ((m <= 64 && n <= 64 && k <= 64)
            || work <= 65_536
            || m <= 4
            || n <= 4
            || k <= 16
            || (work <= 262_144 && (m <= 8 || n <= 8 || k <= 32)));

    if direct {
        let family = if b_row_contiguous && c_row_contiguous && alpha_is_one && beta_is_zero {
            CpuSgemmSeedFamily::DirectAxpyAlpha1Beta0
        } else if b_row_contiguous && c_row_contiguous {
            CpuSgemmSeedFamily::DirectAxpy
        } else {
            CpuSgemmSeedFamily::Direct
        };
        return CpuSgemmSeed {
            family,
            tile_m: 16,
            tile_n: 32,
            tile_k: 16,
            micro_m: 1,
            micro_n: 8,
            pack_a: false,
            pack_b: false,
            persistent_packed_weights: false,
            seed_parallel_2d: false,
        };
    }

    let (tile_m, tile_n, tile_k): (u16, u16, u16) = if m <= 32 || n <= 32 || k <= 32 {
        (32, 64, 64)
    } else if m <= 64 && n >= 512 {
        (32, 128, 128)
    } else if k >= 2_048 && m >= 256 && n >= 256 {
        (64, 128, 192)
    } else if n >= 2_048 && m >= 256 && k >= 256 {
        (64, 192, 128)
    } else if m >= 2_048 && n >= 128 && k >= 128 {
        (96, 128, 128)
    } else if m >= 512 && n >= 512 && k >= 512 && workers >= 4 {
        (64, 128, 192)
    } else {
        (64, 128, 128)
    };

    CpuSgemmSeed {
        family: CpuSgemmSeedFamily::Packed4x8,
        tile_m,
        tile_n,
        tile_k,
        micro_m: 4,
        micro_n: 8,
        pack_a: true,
        pack_b: true,
        persistent_packed_weights: false,
        seed_parallel_2d: workers > 1 && work >= 262_144 && m >= 32 && n >= 32 && k >= 32,
    }
}


/// Optional packed 6 x native-SIMD SGEMM seed recovered from the supplied
/// implementation. It is deliberately emitted in addition to, not instead of,
/// the 4x8 packed seed so offline benchmarking can resolve register pressure,
/// cache behavior and ISA-specific SIMD width on the exact target.
#[inline(always)]
pub fn cpu_sgemm_packed6xsimd_seed(
    m: u64,
    k: u64,
    n: u64,
    workers: u16,
    simd_bytes: u16,
    c_row_contiguous: bool,
) -> Option<CpuSgemmSeed> {
    if !c_row_contiguous || m < 6 || k == 0 || n == 0 || simd_bytes < 16 || simd_bytes % 4 != 0 {
        return None;
    }
    let lanes = simd_bytes / 4;
    if lanes == 0 || n < u64::from(lanes) {
        return None;
    }
    let work = m.saturating_mul(n).saturating_mul(k);
    // Keep the tile exactly divisible by the native SIMD micro-width on every
    // supported fixed/scalable vector target. The multiplier is only a seed;
    // neighboring cache tiles remain eligible for offline enumeration.
    let tile_n = match lanes.checked_mul(8) {
        Some(value) if value != 0 => value,
        _ => return None,
    };
    let (tile_m, tile_k): (u16, u16) = if m <= 64 || n <= 64 || k <= 64 {
        (48, 64)
    } else {
        (96, 128)
    };
    Some(CpuSgemmSeed {
        family: CpuSgemmSeedFamily::Packed6xSimd,
        tile_m,
        tile_n,
        tile_k,
        micro_m: 6,
        micro_n: lanes,
        pack_a: true,
        pack_b: true,
        persistent_packed_weights: false,
        seed_parallel_2d: workers > 1 && work >= 262_144 && m >= 48 && n >= u64::from(lanes),
    })
}

/// Equal-batch small-GEMM candidate from the supplied batched path. The
/// caller supplies the already-canonicalized batch count; broadcasting batches
/// must use other candidates because this source path requires equal prefixes.
#[inline(always)]
pub fn cpu_sgemm_batched_small_seed(
    m: u64,
    k: u64,
    n: u64,
    batch_count: u64,
    workers: u16,
) -> Option<CpuSgemmSeed> {
    if batch_count <= 1 || m == 0 || k == 0 || n == 0 || m > 64 || k > 64 || n > 64 {
        return None;
    }
    let work_per_batch = m.saturating_mul(n).saturating_mul(k);
    Some(CpuSgemmSeed {
        family: CpuSgemmSeedFamily::BatchedSmall,
        tile_m: m.min(u64::from(u16::MAX)) as u16,
        tile_n: n.min(u64::from(u16::MAX)) as u16,
        tile_k: k.min(u64::from(u16::MAX)) as u16,
        micro_m: 1,
        micro_n: 8,
        pack_a: false,
        pack_b: false,
        persistent_packed_weights: false,
        // Mirrors the source gate: parallelize by batch only when there are
        // enough batches and each tiny GEMM has enough arithmetic to amortize.
        seed_parallel_2d: workers > 1
            && batch_count >= u64::from(workers).saturating_mul(2)
            && work_per_batch >= 512,
    })
}

/// Fixed-weight SGEMM seed derived from the supplied `PackedSgemmB` path.
/// Packing is a one-time binding/load action, never replay-time autotuning.
#[inline(always)]
pub fn cpu_sgemm_prepacked_b_seed(
    m: u64,
    k: u64,
    n: u64,
    workers: u16,
) -> CpuSgemmSeed {
    let (tile_m, tile_n, tile_k): (u16, u16, u16) = if m <= 32 || n <= 32 || k <= 32 {
        (32, 64, 64)
    } else if m <= 64 && n >= 512 {
        (32, 128, 128)
    } else if k >= 2_048 && m >= 256 && n >= 256 {
        (64, 128, 192)
    } else if n >= 2_048 && m >= 256 && k >= 256 {
        (64, 192, 128)
    } else if m >= 2_048 && n >= 128 && k >= 128 {
        (96, 128, 128)
    } else if m >= 512 && n >= 512 && k >= 512 && workers >= 4 {
        (64, 128, 192)
    } else {
        (64, 128, 128)
    };
    let tile_m_u64 = u64::from(tile_m);
    let row_blocks = m.div_ceil(tile_m_u64);
    CpuSgemmSeed {
        family: CpuSgemmSeedFamily::PrepackedB4x8,
        tile_m,
        tile_n,
        tile_k,
        micro_m: 4,
        micro_n: 8,
        pack_a: true,
        pack_b: true,
        persistent_packed_weights: true,
        seed_parallel_2d: workers > 1 && row_blocks > 1 && m >= 128 && n >= 64 && k >= 64,
    }
}

/// Rank-1 matmul specialization recovered from the supplied `linalg.mojo`.
/// This is only an offline enumeration seed; replay performs no shape dispatch.
#[inline(always)]
pub const fn cpu_vector_matmul_kernel_family_seed(
    lhs_rank: usize,
    rhs_rank: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    rhs_row_contiguous: bool,
) -> Option<KernelFamily> {
    if transpose_lhs || transpose_rhs {
        return None;
    }
    if lhs_rank == 1 && rhs_rank == 1 {
        Some(KernelFamily::CpuDotSimd)
    } else if lhs_rank == 1 && rhs_rank == 2 {
        if rhs_row_contiguous {
            Some(KernelFamily::CpuVecMatAxpy)
        } else {
            Some(KernelFamily::CpuVecMatSimd)
        }
    } else if lhs_rank == 2 && rhs_rank == 1 {
        Some(KernelFamily::CpuMatVecSimd)
    } else {
        None
    }
}

/// Source-derived parallelization hint for vector/matrix products. Large output
/// dimensions are statically chunked only when the reduction length is large
/// enough to amortize worker scheduling.
#[inline(always)]
pub const fn cpu_vector_matmul_parallel_seed(
    output_elements: u64,
    reduction_len: u64,
    workers: u16,
) -> bool {
    workers > 1 && output_elements >= 4_096 && reduction_len >= 16
}

/// Primary exact 1-D FFT algorithm seed from the supplied radix-2/Bluestein
/// implementation. Arbitrary-size alternatives may still be enumerated and
/// benchmarked offline.
#[inline(always)]
pub const fn fft_kernel_family_seed(
    transform_len: u64,
    axis_is_last: bool,
    gpu_candidate: bool,
) -> KernelFamily {
    if gpu_candidate && axis_is_last && (transform_len == 2 || transform_len == 4) {
        KernelFamily::FftSmallLastDimGpu
    } else if transform_len > 0 && (transform_len & (transform_len - 1)) == 0 {
        KernelFamily::FftRadix2
    } else {
        KernelFamily::FftBluestein
    }
}

/// Reuse-aware gate for enumerating the fixed-weight prepacked-B SGEMM path.
/// The decision is made offline from the canonical reuse contract.
#[inline(always)]
pub const fn cpu_should_seed_persistent_weight_prepack(
    reuse: ReusePattern,
    policy: CpuOfflineSeedPolicy,
) -> bool {
    if !policy.seed_persistent_weight_prepack {
        return false;
    }
    match reuse {
        ReusePattern::PersistentWeight => true,
        _ => false,
    }
}

/// Dense linear-algebra seed. The supplied source has dedicated exact small
/// kernels for solve/inverse/Cholesky at 2x2 and 3x3; larger/batched problems
/// seed the generic statically-sharded family. VendorExact should still be
/// included by candidate generation when an exact provider binding is possible.
#[inline(always)]
pub const fn dense_linalg_kernel_family_seed(
    kind: DenseLinalgKind,
    n: u64,
    batch_count: u64,
) -> KernelFamily {
    let supports_small_static = match kind {
        DenseLinalgKind::Solve | DenseLinalgKind::Inverse | DenseLinalgKind::CholeskyLower => true,
    };
    if batch_count == 1 && (n == 2 || n == 3) && supports_small_static {
        KernelFamily::CpuLinalgSmallStatic
    } else {
        KernelFamily::CpuLinalgBatched
    }
}

/// Offline seed for native warp/wavefront sizes 32 (CUDA and many HIP targets)
/// or 64 (some AMD HIP targets). No candidate is produced for an unsupported
/// warp size, insufficient hardware limit, or an insufficient policy limit.
/// A limit MUST be rounded down, never up, or the seed could violate its guard.
/// Zero means an infeasible candidate family, NOT an executable zero-thread block.
#[inline(always)]
fn gpu_warp_seed_cap(
    device_limit: u16,
    warp_size: u8,
    requested_cap: u16,
    default_cap: u16,
) -> u16 {
    let warp = u16::from(warp_size);
    if warp != 32 && warp != 64 {
        return 0;
    }
    let requested = if requested_cap == 0 { default_cap } else { requested_cap };
    let cap = requested.min(device_limit).min(1024);
    cap - cap % warp
}

#[inline(always)]
pub fn gpu_rowwise_threads_seed(
    hw: &GpuHardwareContract,
    row_n: u64,
    requested_cap: u16,
    policy: GpuOfflineSeedPolicy,
) -> u16 {
    let cap = gpu_warp_seed_cap(
        hw.max_threads_per_block, hw.warp_size,
        requested_cap, policy.default_threads_per_block,
    );
    if cap == 0 { return 0; }
    if cap >= 1024 && row_n > 512 {
        1024
    } else if cap >= 512 && row_n > 256 {
        512
    } else if cap >= 256 && row_n > 128 {
        256
    } else if cap >= 128 && row_n > 64 {
        128
    } else if cap >= 64 && row_n > 32 {
        64
    } else {
        u16::from(hw.warp_size)
    }
}

#[inline(always)]
pub fn gpu_elementwise_threads_seed(
    hw: &GpuHardwareContract,
    n: u64,
    cost: GpuElementwiseCostClass,
    requested_cap: u16,
    policy: GpuOfflineSeedPolicy,
) -> u16 {
    let cap = gpu_warp_seed_cap(
        hw.max_threads_per_block, hw.warp_size,
        requested_cap, policy.default_threads_per_block,
    );
    if cap == 0 { return 0; }
    if n < 4_096 {
        return cap.min(128);
    }
    if cost == GpuElementwiseCostClass::Expensive {
        return cap.min(256);
    }
    if n >= 1_048_576 && cap >= 512 {
        512
    } else {
        cap
    }
}

/// Zero means a frozen empty/no-launch stage, or an invalid seed cap.  It is
/// never a license to launch a zero-thread kernel or silently replace the
/// published launch policy. All positive block counts respect the policy cap.
#[inline(always)]
pub fn gpu_elementwise_blocks_seed(n: u64, threads: u16, policy: GpuOfflineSeedPolicy) -> u32 {
    if n == 0 || threads == 0 || policy.max_elementwise_blocks == 0 {
        return 0;
    }
    let blocks = n.div_ceil(u64::from(threads));
    // The cap is u32, so narrowing after min is provably lossless.
    blocks.min(u64::from(policy.max_elementwise_blocks)) as u32
}

#[cfg(test)]
#[path = "../../tests/tensor_light/unit/aot/ranking_regressions.rs"]
mod ranking_regressions;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/aot/seed_arithmetic_regressions.rs"]
mod seed_arithmetic_regressions;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/aot/offline_target_and_policy_regressions.rs"]
mod offline_target_and_policy_regressions;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/aot/fallible_candidate_memory.rs"]
mod fallible_candidate_memory;
#[cfg(test)]
#[path = "../../tests/tensor_light/unit/aot/cpu_core_set_worker_seed.rs"]
mod cpu_core_set_worker_seed;
