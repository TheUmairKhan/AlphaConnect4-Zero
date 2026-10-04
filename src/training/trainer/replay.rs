use std::collections::VecDeque;

use burn::prelude::*;
use rand::{Rng, RngExt, seq::index};

use crate::training::self_play::self_play::TrainingExample;

pub struct ReplayBuffer {
    games: VecDeque<Vec<TrainingExample>>,
    capacity_games: usize,
    positions: usize,
}

impl ReplayBuffer {
    pub fn new(capacity_games: usize) -> Self {
        Self {
            games: VecDeque::with_capacity(capacity_games),
            capacity_games,
            positions: 0,
        }
    }

    pub fn push(&mut self, game: Vec<TrainingExample>) {
        if self.games.len() == self.capacity_games {
            let oldest = self.games.pop_front().unwrap();
            self.positions -= oldest.len();
        }
        self.positions += game.len();
        self.games.push_back(game);
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn sample_examples(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<TrainingExample> {
        let game_ends: Vec<usize> = self.games.iter().scan(0, |end, game| {
            *end += game.len();
            Some(*end)
        }).collect();

        index::sample(rng, self.positions, batch_size).iter().map(|index| {
            let game_idx = game_ends.partition_point(|&end| end <= index);
            let game_start = if game_idx == 0 { 0 } else { game_ends[game_idx - 1] };
            let example = &self.games[game_idx][index - game_start];
            if rng.random_bool(0.5) { example.mirrored() } else { example.clone() }
        }).collect()
    }
}

pub fn batch_tensors<B: Backend>(examples: &[TrainingExample], device: &B::Device)
    -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>)
{
    let batch_size = examples.len();
    let mut states = Vec::with_capacity(batch_size * 2 * 6 * 7);
    let mut policies = Vec::with_capacity(batch_size * 7);
    let mut values = Vec::with_capacity(batch_size);
    for example in examples {
        states.extend(example.state.iter().flatten().flatten().copied());
        policies.extend_from_slice(&example.policy);
        values.push(example.value);
    }

    (
        Tensor::from_data(TensorData::new(states, [batch_size, 2, 6, 7]), device),
        Tensor::from_data(TensorData::new(policies, [batch_size, 7]), device),
        Tensor::from_data(TensorData::new(values, [batch_size, 1]), device),
    )
}
