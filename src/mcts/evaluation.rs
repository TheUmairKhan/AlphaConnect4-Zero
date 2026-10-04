use std::{
    cell::{Cell, RefCell}, rc::Rc, sync::mpsc::SyncSender, time::Instant,
};

use burn::prelude::Backend;
use futures::channel::oneshot;

use crate::board::Board;

use super::policy::{NetworkTiming, ZeroNetPolicy};

#[derive(Clone, Copy, Debug)]
pub struct EvalOutput {
    pub priors: [f32; 7],
    pub value: f32,
}

pub trait LeafEvaluator {
    async fn evaluate(&self, board: Board) -> EvalOutput;
    fn timing(&self) -> NetworkTiming;
}

impl<E: LeafEvaluator> LeafEvaluator for &E {
    async fn evaluate(&self, board: Board) -> EvalOutput {
        (**self).evaluate(board).await
    }

    fn timing(&self) -> NetworkTiming {
        (**self).timing()
    }
}

pub struct BatchRequest {
    pub boards: Vec<Board>,
    pub sender: SyncSender<Vec<EvalOutput>>,
}

pub enum WorkerMessage {
    Batch(BatchRequest),
    Finished,
}
pub struct BatchingClient {
    queue: Rc<RefCell<Vec<(Board, oneshot::Sender<EvalOutput>)>>>,
    timing: Cell<NetworkTiming>,
}

impl BatchingClient {
    pub fn new(queue: Rc<RefCell<Vec<(Board, oneshot::Sender<EvalOutput>)>>>) -> Self {
        Self {
            queue,
            timing: Cell::new(NetworkTiming::default()),
        }
    }
}

impl LeafEvaluator for BatchingClient {
    async fn evaluate(&self, board: Board) -> EvalOutput {
        let started = Instant::now();
        let (sender, receiver) = oneshot::channel();
        self.queue.borrow_mut().push((board, sender));
        let output = receiver.await.expect("batch dropped");

        let mut timing = self.timing.get();
        timing.calls += 1;
        timing.total += started.elapsed();
        self.timing.set(timing);

        output
    }

    fn timing(&self) -> NetworkTiming { self.timing.get() }
}

impl<B: Backend> LeafEvaluator for ZeroNetPolicy<B> {
    async fn evaluate(&self, board: Board) -> EvalOutput {
        ZeroNetPolicy::evaluate(self, std::slice::from_ref(&board)).pop().unwrap()
    }

    fn timing(&self) -> NetworkTiming {
        ZeroNetPolicy::timing(self)
    }
}
