use std::{
    cell::Cell,
    sync::mpsc::{self, SyncSender},
    time::Instant,
};

use burn::prelude::Backend;

use crate::board::Board;

use super::policy::{NetworkTiming, ZeroNetPolicy};

#[derive(Clone, Copy, Debug)]
pub struct EvalOutput {
    pub priors: [f32; 7],
    pub value: f32,
}

pub trait LeafEvaluator {
    fn evaluate(&self, board: Board) -> EvalOutput;
    fn timing(&self) -> NetworkTiming;
}

pub struct LeafRequest {
    pub board: Board,
    pub sender: SyncSender<EvalOutput>,
}

#[derive(Clone)]
pub struct EvalClient {
    requests: SyncSender<LeafRequest>,
    timing: Cell<NetworkTiming>,
}

impl EvalClient {
    pub fn new(requests: SyncSender<LeafRequest>) -> Self {
        Self {
            requests,
            timing: Cell::new(NetworkTiming::default()),
        }
    }
}

impl LeafEvaluator for EvalClient {
    fn evaluate(&self, board: Board) -> EvalOutput {
        let started = Instant::now();
        let (sender, receiver) = mpsc::sync_channel(1);
        self.requests
            .send(LeafRequest { board, sender })
            .expect("evaluation request queue closed");
        let output = receiver.recv().expect("evaluation reply channel closed");

        // For a client, total includes queue wait and evaluation, not just model execution.
        let mut timing = self.timing.get();
        timing.calls += 1;
        timing.total += started.elapsed();
        self.timing.set(timing);
        output
    }

    fn timing(&self) -> NetworkTiming {
        self.timing.get()
    }
}

impl<B: Backend> LeafEvaluator for ZeroNetPolicy<B> {
    fn evaluate(&self, board: Board) -> EvalOutput {
        ZeroNetPolicy::evaluate(self, std::slice::from_ref(&board)).pop().unwrap()
    }

    fn timing(&self) -> NetworkTiming {
        ZeroNetPolicy::timing(self)
    }
}
