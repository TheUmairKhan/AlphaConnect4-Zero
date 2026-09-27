use burn::prelude::*;

use crate::{board::Board, model::zeronet::ZeroNet};

pub struct ZeroNetPolicy<B: Backend> {
    model: ZeroNet<B>,
    device: B::Device,
}

impl<B: Backend> ZeroNetPolicy<B> {
    pub fn new(model: ZeroNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    pub fn evaluate(&self, state: &Board) -> ([f32; 7], f32) {
        let input = Tensor::<B, 4>::from_data([state.encode_state()], &self.device);
        let (policy_logits, value) = self.model.forward(input);
        let logits: Vec<f32> = policy_logits.into_data().iter::<f32>().collect();
        let valid_moves = state.valid_moves();
        let max_logit = valid_moves
            .iter()
            .map(|&action| logits[action])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut priors = [0.0; 7];
        let mut total = 0.0;

        for action in valid_moves {
            let prior = (logits[action] - max_logit).exp();
            priors[action] = prior;
            total += prior;
        }

        for prior in &mut priors {
            *prior /= total;
        }

        let value = value.into_data().iter::<f32>().next().unwrap();
        
        (priors, value)
    }
}
