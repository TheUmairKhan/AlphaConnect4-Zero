use std::{error::Error, fs, path::Path};

use serde::Deserialize;

#[derive(Deserialize)]
pub struct TrainingConfig {
    pub self_play: SelfPlayConfig,
}

#[derive(Deserialize)]
pub struct SelfPlayConfig {
    pub simulations: u32,
    pub c_puct: f32,
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
        Ok(config)
    }
}

impl SelfPlayConfig {
    pub fn temperature(&self, move_number: usize) -> f32 {
        let progress = (move_number as f32 / self.temperature_decay_moves as f32).min(1.0);
        self.temperature_start + (self.temperature_end - self.temperature_start) * progress
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.simulations == 0 || !self.c_puct.is_finite() || self.c_puct <= 0.0 {
            return Err("simulations and c_puct must be positive".into());
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