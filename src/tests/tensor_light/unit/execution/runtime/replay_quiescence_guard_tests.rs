// SPDX-FileCopyrightText: 2026 Morteza Taleblou
// SPDX-FileCopyrightText: 2026 Mitra Daneshmand
// SPDX-License-Identifier: MIT

use super::ReplayQuiescenceGuard;
use crate::tensor_light::execution::{
    BoundBackend, CpuGpuSlice, CpuKernelPlan, FrozenNode, GpuGraphSegmentPlan, GpuKernelPlan,
    HostBoundaryClass, SyncKind, SyncPlan, TransferBatchPlan, TransferPlan,
};

#[derive(Clone, Copy)]
enum Outcome {
    Success,
    Error,
    Panic,
}

struct RecordingBackend {
    outcome: Outcome,
    dispatches: usize,
    fences: usize,
    pending: bool,
}

// SAFETY: This fixture holds no borrowed buffers, performs no native work,
// and clears its modeled pending state synchronously without panicking.
unsafe impl BoundBackend for RecordingBackend {
    type Error = &'static str;

    fn execute_cpu_node(
        &mut self,
        _: &'static FrozenNode,
        _: &'static CpuKernelPlan,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_gpu_node(
        &mut self,
        _: &'static FrozenNode,
        _: &'static GpuKernelPlan,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn launch_cpu_slice(
        &mut self,
        _: &'static FrozenNode,
        _: u16,
        _: &'static CpuGpuSlice,
        _: &'static CpuKernelPlan,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn launch_gpu_slice(
        &mut self,
        _: &'static FrozenNode,
        _: u16,
        _: &'static CpuGpuSlice,
        _: &'static GpuKernelPlan,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn complete_cpu_gpu_slice(
        &mut self,
        _: &'static FrozenNode,
        _: u16,
        _: &'static CpuGpuSlice,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_gpu_graph_segment(
        &mut self,
        _: &'static GpuGraphSegmentPlan,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_transfer(&mut self, _: &'static TransferPlan) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_host_readback(
        &mut self,
        _: &'static TransferPlan,
        _: HostBoundaryClass,
    ) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_transfer_batch(&mut self, _: &'static TransferBatchPlan) -> Result<(), Self::Error> {
        unreachable!("only sync dispatch is exercised")
    }

    fn execute_sync(&mut self, _: &'static SyncPlan) -> Result<(), Self::Error> {
        self.dispatches += 1;
        self.pending = true;
        match self.outcome {
            Outcome::Success => Ok(()),
            Outcome::Error => Err("dispatch failed"),
            Outcome::Panic => panic!("dispatch panic"),
        }
    }

    fn quiesce_or_abort(&mut self) {
        self.pending = false;
        self.fences = self.fences.saturating_add(1);
    }
}

static SYNC: SyncPlan = SyncPlan {
    id: 0,
    kind: SyncKind::CpuBarrier,
    event: None,
    barrier: Some(0),
    producer_node: 0,
    producer_slice: None,
    consumer_node: 1,
    consumer_slice: None,
    stream: None,
};

fn guarded_dispatch(backend: &mut RecordingBackend) -> Result<(), &'static str> {
    let guard = ReplayQuiescenceGuard { backend };
    guard.backend.execute_sync(&SYNC)
}

#[test]
fn guard_establishes_quiescence_on_success_and_early_backend_error() {
    let mut backend = RecordingBackend {
        outcome: Outcome::Success,
        dispatches: 0,
        fences: 0,
        pending: false,
    };
    assert_eq!(guarded_dispatch(&mut backend), Ok(()));
    assert_eq!(
        (backend.dispatches, backend.fences, backend.pending),
        (1, 1, false)
    );

    backend.outcome = Outcome::Error;
    assert_eq!(guarded_dispatch(&mut backend), Err("dispatch failed"));
    assert_eq!(
        (backend.dispatches, backend.fences, backend.pending),
        (2, 2, false)
    );
}

#[cfg(panic = "unwind")]
#[test]
fn guard_fences_before_replay_unwind_reaches_the_caller() {
    let mut backend = RecordingBackend {
        outcome: Outcome::Panic,
        dispatches: 0,
        fences: 0,
        pending: false,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        guarded_dispatch(&mut backend)
    }));
    assert!(result.is_err());
    assert_eq!(
        (backend.dispatches, backend.fences, backend.pending),
        (1, 1, false)
    );
}
