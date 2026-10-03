mod board;
mod config;
mod env;
mod mcts;
mod metrics;
pub mod model;
mod training;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = burn::backend::Autodiff<burn::backend::Metal<f32>>;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "training.toml".to_string());
    let config = config::TrainingConfig::load(path)?;
    training::trainer::trainer::run::<Backend>(&config, &Default::default())
}
