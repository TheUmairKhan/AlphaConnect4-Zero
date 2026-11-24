mod board;
mod config;
mod env;
mod mcts;
mod metrics;
pub mod model;
mod play;
mod training;

#[cfg(feature = "cuda")]
type Backend = burn::backend::Cuda<f32>;
#[cfg(all(feature = "cpu", not(feature = "cuda")))]
type Backend = burn::backend::NdArray<f32>;
#[cfg(all(not(feature = "cuda"), not(feature = "cpu"), target_os = "macos"))]
type Backend = burn::backend::Metal<f32>;
#[cfg(all(not(feature = "cuda"), not(feature = "cpu"), not(target_os = "macos")))]
type Backend = burn::backend::Wgpu<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
