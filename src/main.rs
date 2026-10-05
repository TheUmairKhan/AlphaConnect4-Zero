mod board;
mod config;
mod env;
mod mcts;
mod metrics;
pub mod model;
mod play;
mod training;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = burn::backend::Metal<f32>;

    let mut args = std::env::args().skip(1).peekable();
    let playing = args.next_if(|arg| arg == "play").is_some();
    let path = args.next().unwrap_or_else(|| "training.toml".to_string());
    let config = config::TrainingConfig::load(path)?;
    if playing {
        play::run::<Backend>(&config, &Default::default())
    } else {
        training::trainer::trainer::run::<burn::backend::Autodiff<Backend>>(&config, &Default::default())
    }
}
