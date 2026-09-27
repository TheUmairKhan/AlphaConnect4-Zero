use std::{cell::Cell, time::{Duration, Instant}};

use burn::prelude::*;

use crate::{board::Board, model::zeronet::ZeroNet};

#[derive(Clone, Copy, Default)]
pub struct NetworkTiming {
    pub calls: usize,
    pub input: Duration,
    pub forward: Duration,
    pub readback: Duration,
    pub total: Duration,
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

    pub fn evaluate(&self, state: &Board) -> ([f32; 7], f32) {
        let started = Instant::now();
        let input = Tensor::<B, 4>::from_data([state.encode_state()], &self.device);
        let input_done = Instant::now();
        let (policy_logits, value) = self.model.forward(input);
        let forward_done = Instant::now();
        let logits: Vec<f32> = policy_logits.into_data().iter::<f32>().collect();
        let value_data = value.into_data();
        let readback_done = Instant::now();
        let valid_moves = state.valid_moves();
        let max_logit = valid_moves
            .iter()
            .map(|&action| logits[action as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut priors = [0.0; 7];
        let mut total = 0.0;

        for action in valid_moves {
            let action = action as usize;
            let prior = (logits[action] - max_logit).exp();
            priors[action] = prior;
            total += prior;
        }

        for prior in &mut priors {
            *prior /= total;
        }

        let value = value_data.iter::<f32>().next().unwrap();
        let mut timing = self.timing.get();
        timing.calls += 1;
        timing.input += input_done.duration_since(started);
        timing.forward += forward_done.duration_since(input_done);
        timing.readback += readback_done.duration_since(forward_done);
        timing.total += started.elapsed();
        self.timing.set(timing);
        
        (priors, value)
    }
}
