use std::{cell::Cell, time::{Duration, Instant}};

use burn::{prelude::*, tensor::Transaction};

use crate::{board::Board, model::zeronet::ZeroNet};

use super::evaluation::EvalOutput;

#[derive(Clone, Copy, Default)]
pub struct NetworkTiming {
    pub calls: usize,
    pub input: Duration,
    pub forward: Duration,
    pub readback: Duration,
    pub total: Duration,
}

impl NetworkTiming {
    pub fn since(self, earlier: Self) -> Self {
        Self {
            calls: self.calls - earlier.calls,
            input: self.input - earlier.input,
            forward: self.forward - earlier.forward,
            readback: self.readback - earlier.readback,
            total: self.total - earlier.total,
        }
    }
}

pub struct ZeroNetPolicy<B: Backend> {
    model: ZeroNet<B>,
    device: B::Device,
    timing: Cell<NetworkTiming>,
}

impl<B: Backend> ZeroNetPolicy<B> {
    pub fn new(model: ZeroNet<B>, device: B::Device) -> Self {
        Self { model, device, timing: Cell::new(NetworkTiming::default()) }
    }

    pub fn timing(&self) -> NetworkTiming {
        self.timing.get()
    }

    pub fn evaluate(&self, boards: &[Board]) -> Vec<EvalOutput> {
        let started = Instant::now();
        let mut states = Vec::with_capacity(boards.len() * 2 * 6 * 7);
        for board in boards {
            states.extend(board.encode_state().into_iter().flatten().flatten());
        }
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(states, [boards.len(), 2, 6, 7]),
            &self.device,
        );
        let input_done = Instant::now();
        let (policy_logits, value) = self.model.forward(input);
        let forward_done = Instant::now();
        let data = Transaction::default().register(policy_logits).register(value).execute();
        let logits: Vec<f32> = data[0].iter::<f32>().collect();
        let values: Vec<f32> = data[1].iter::<f32>().collect();
        let readback_done = Instant::now();

        let outputs = boards.iter().enumerate().map(|(index, board)| {
            let row = &logits[index * 7..(index + 1) * 7];
            let valid_moves = board.valid_moves();
            let max_logit = valid_moves
                .iter()
                .map(|&action| row[action as usize])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut priors = [0.0; 7];
            let mut total = 0.0;

            for action in valid_moves {
                let action = action as usize;
                let prior = (row[action] - max_logit).exp();
                priors[action] = prior;
                total += prior;
            }

            for prior in &mut priors {
                *prior /= total;
            }
            EvalOutput { priors, value: values[index] }
        }).collect();

        let mut timing = self.timing.get();
        timing.calls += 1;
        timing.input += input_done.duration_since(started);
        timing.forward += forward_done.duration_since(input_done);
        timing.readback += readback_done.duration_since(forward_done);
        timing.total += started.elapsed();
        self.timing.set(timing);
        
        outputs
    }
}
