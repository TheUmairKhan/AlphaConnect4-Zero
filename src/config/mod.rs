use std::{error::Error, fs, path::Path};

use serde::Deserialize;

#[derive(Deserialize)]
pub struct TrainingConfig {
    pub search: SearchConfig,
    pub self_play: SelfPlayConfig,
    pub model: ModelConfig,
    pub training: TrainerConfig,
    pub evaluation: EvaluationConfig,
}

#[derive(Deserialize)]
pub struct SearchConfig {
    pub c_puct: f32,
}

#[derive(Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_res_blocks: usize,
    pub initial_checkpoint: Option<String>,
}

#[derive(Deserialize)]
pub struct TrainerConfig {
    pub total_updates: usize,
    pub batch_size: usize,
    pub replay_capacity_games: usize,
    pub learning_rate: f64,
    pub l2_coefficient: f64,
    pub checkpoint_every_updates: usize,
    pub checkpoint_dir: String,
}

#[derive(Deserialize)]
pub struct EvaluationConfig {
    pub threads: usize,
    pub every_updates: usize,
    pub depths: Vec<u32>,
    pub games: usize,
    pub simulations: u32,
}

#[derive(Deserialize)]
pub struct SelfPlayConfig {
    pub threads: usize,
    pub games_per_thread: usize,
    pub model_refresh_per_updates: usize,
    pub simulations: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub temperature_decay_moves: usize,
}

impl TrainingConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.self_play.validate()?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        let train = &self.training;
        if !self.search.c_puct.is_finite() || self.search.c_puct <= 0.0 {
            return Err("c_puct must be positive".into());
        }
        if self.model.hidden_size == 0 || self.model.num_res_blocks == 0 {
            return Err("model dimensions must be positive".into());
        }
        if train.total_updates == 0
            || train.batch_size == 0
            || train.replay_capacity_games == 0
            || train.checkpoint_every_updates == 0
            || train.checkpoint_dir.is_empty()
            || !train.learning_rate.is_finite()
            || train.learning_rate <= 0.0
            || !train.l2_coefficient.is_finite()
            || train.l2_coefficient < 0.0
        {
            return Err("invalid training settings".into());
        }
        if train.replay_capacity_games.saturating_mul(7) < train.batch_size {
            return Err("replay_capacity_games is too small to hold one batch".into());
        }
        let evaluation = &self.evaluation;
        if evaluation.threads == 0
            || evaluation.every_updates == 0
            || evaluation.simulations == 0
        {
            return Err("evaluation threads, every_updates, and simulations must be positive".into());
        }
        if evaluation.games < 2 {
            return Err("evaluation games must be at least 2 so the model plays both first and second".into());
        }
        if evaluation.depths.is_empty() || evaluation.depths.contains(&0) {
            return Err("evaluation depths must be a non-empty list of positive depths".into());
        }
        Ok(())
    }
}

impl SelfPlayConfig {
    pub fn temperature(&self, move_number: usize) -> f32 {
        let progress = (move_number as f32 / self.temperature_decay_moves as f32).min(1.0);
        self.temperature_start + (self.temperature_end - self.temperature_start) * progress
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.simulations == 0 || self.threads == 0 || self.games_per_thread == 0 || self.model_refresh_per_updates == 0 {
            return Err("simulations, threads, games_per_thread, and model_refresh_per_updates must be positive".into());
        }
        if !self.dirichlet_alpha.is_finite() || self.dirichlet_alpha <= 0.0 {
            return Err("dirichlet_alpha must be positive".into());
        }
        if !self.dirichlet_epsilon.is_finite()
            || !(0.0..=1.0).contains(&self.dirichlet_epsilon)
        {
            return Err("dirichlet_epsilon must be between 0 and 1".into());
        }
        if !self.temperature_start.is_finite()
            || self.temperature_start <= 0.0
            || !self.temperature_end.is_finite()
            || self.temperature_end <= 0.0
            || self.temperature_decay_moves == 0
        {
            return Err("temperatures and temperature_decay_moves must be positive".into());
        }
        Ok(())
    }
}
