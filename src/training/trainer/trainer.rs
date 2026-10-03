use std::{
    error::Error,
    fs,
    path::Path,
    process,
    sync::{Condvar, Mutex, atomic::{AtomicUsize, Ordering}},
    thread,
    time::Instant,
};

use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::DefaultRecorder,
    tensor::{activation::log_softmax, backend::AutodiffBackend},
};

use crate::{
    config::TrainingConfig,
    mcts::policy::ZeroNetPolicy,
    metrics::{
        metrics::{MetricsLogger, StepMetrics},
        performance::{TimingLogger, UpdateTiming},
    },
    model::zeronet::{ZeroNet, ZeroNetConfig},
    training::{
        accelerator::accelerator,
        evaluation::arena,
        self_play::generator::generate_self_play_games,
    },
};

use super::{
    l2::squared_l2,
    replay::{ReplayBuffer, batch_tensors},
};

pub fn run<B: AutodiffBackend>(config: &TrainingConfig, device: &B::Device) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut model = ZeroNetConfig::new(2, config.model.hidden_size, config.model.num_res_blocks).init::<B>(device);
    let initial_model = model.valid();
    let mut optimizer = AdamConfig::new().init::<B, ZeroNet<B>>();
    let replay = (Mutex::new(ReplayBuffer::new(config.training.replay_capacity_games)), Condvar::new());
    let pending_model = Mutex::new(None::<(usize, ZeroNet<B::InnerBackend>)>);
    let self_play_gate = Mutex::new(());
    let games_completed = AtomicUsize::new(0);
    let positions_generated = AtomicUsize::new(0);
    let mut rng = rand::rng();
    let checkpoint_dir = Path::new(&config.training.checkpoint_dir);
    fs::create_dir_all(checkpoint_dir)?;
    let mut metrics_logger = MetricsLogger::new(checkpoint_dir)?;
    let timing_logger = Mutex::new(TimingLogger::new(checkpoint_dir)?);
    println!("Live metrics: {}", checkpoint_dir.join("metrics.html").display());
    println!("Performance: {}", checkpoint_dir.join("performance.html").display());

    thread::scope(|scope| {
        scope.spawn(|| {
            let mut policy = ZeroNetPolicy::new(initial_model, device.clone());
            let mut model_step = 0;
            generate_self_play_games(
                config,
                |request_receiver| accelerator::serve_requests(
                    request_receiver,
                    config.self_play.threads,
                    |boards| {
                        let _running = self_play_gate.lock().unwrap();
                        if let Some((step, latest)) = pending_model.lock().unwrap().take() {
                            policy = ZeroNetPolicy::new(latest, device.clone());
                            model_step = step;
                            println!("evaluator switched to model from update {step}");
                        }
                        let before = policy.timing();
                        let outputs = policy.evaluate(boards);
                        let batch_timing = policy.timing().since(before);
                        if let Err(error) = timing_logger.lock().unwrap()
                            .accelerator_batch(model_step, boards.len(), batch_timing)
                        {
                            eprintln!("failed to log accelerator batch: {error}");
                            process::exit(1);
                        }
                        outputs
                    },
                ),
                |game_number, game| {
                    let positions = game.examples.len();
                    timing_logger.lock().unwrap()
                        .game(game_number, &game.timing, config.self_play.simulations, positions)
                        .expect("failed to log self-play game");
                    let (buffer, ready) = &replay;
                    buffer.lock().unwrap().push(game.examples);
                    positions_generated.fetch_add(positions, Ordering::Relaxed);
                    games_completed.fetch_add(1, Ordering::Relaxed);
                    ready.notify_one();
                },
            );
        });

        let training_result = (|| -> Result<(), Box<dyn Error>> {
            for step in 1..=config.training.total_updates {
                let update_started = Instant::now();
                let examples = {
                    let (buffer, ready) = &replay;
                    let mut buffer = buffer.lock().unwrap();
                    while buffer.positions() < config.training.batch_size {
                        buffer = ready.wait(buffer).unwrap();
                    }
                    buffer.sample_examples(config.training.batch_size, &mut rng)
                };
                let (states, policies, values) = batch_tensors::<B>(&examples, device);
                let sample_elapsed = update_started.elapsed();
                let forward_started = Instant::now();
                let (logits, predictions) = model.forward(states);
                let forward_elapsed = forward_started.elapsed();
                let loss_started = Instant::now();
                let log_policy = log_softmax(logits, 1);
                let policy_loss = -(policies * log_policy.clone()).sum_dim(1).mean();
                let error = predictions - values;
                let value_loss = (error.clone() * error.clone()).mean();
                let metrics = StepMetrics::from_batch(
                    step, &policy_loss, &value_loss, log_policy, error,
                );
                let loss = policy_loss + value_loss
                    + squared_l2(&model, device) * config.training.l2_coefficient;
                let loss_elapsed = loss_started.elapsed();
                let backward_started = Instant::now();
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                let backward_elapsed = backward_started.elapsed();
                let optimizer_started = Instant::now();
                model = optimizer.step(config.training.learning_rate, model, gradients);
                let optimizer_elapsed = optimizer_started.elapsed();
                let logging_started = Instant::now();
                metrics_logger.record(metrics)?;
                let logging_elapsed = logging_started.elapsed();
                timing_logger.lock().unwrap().update(step, UpdateTiming {
                    total: update_started.elapsed(),
                    sample: sample_elapsed,
                    forward: forward_elapsed,
                    loss_metrics: loss_elapsed,
                    backward: backward_elapsed,
                    optimizer: optimizer_elapsed,
                    logging: logging_elapsed,
                })?;

                if step % config.self_play.model_refresh_per_updates == 0 {
                    *pending_model.lock().unwrap() = Some((step, model.valid()));
                    println!("published evaluator model at update {step}");
                }
                if step % config.evaluation.every_updates == 0 {
                    let _paused = self_play_gate.lock().unwrap();
                    println!("evaluating model from update {step} against minimax depths {:?}", config.evaluation.depths);
                    let evaluation_started = Instant::now();
                    let policy = ZeroNetPolicy::new(model.valid(), device.clone());
                    let summaries = arena::evaluate(&policy, config);
                    for summary in &summaries {
                        println!(
                            "  depth {}: {}W/{}L/{}D, score {:.3}, elo {:+.0} (first {:.3}, second {:.3})",
                            summary.depth, summary.wins, summary.losses, summary.draws, summary.score_rate(),
                            summary.elo(), summary.first_player_score_rate(), summary.second_player_score_rate(),
                        );
                    }
                    metrics_logger.record_evaluation(step, &summaries)?;
                    println!("evaluation finished in {:.1}s", evaluation_started.elapsed().as_secs_f32());
                }
                if step % config.training.checkpoint_every_updates == 0
                    || step == config.training.total_updates
                {
                    model.valid().save_file(
                        checkpoint_dir.join(format!("model-step-{step}")),
                        &DefaultRecorder::new(),
                    )?;
                }
            }
            Ok(())
        })();

        if let Err(error) = training_result {
            eprintln!("training failed: {error}");
            process::exit(1);
        }
        let games = games_completed.load(Ordering::Relaxed);
        let positions = positions_generated.load(Ordering::Relaxed);
        let replay_games = replay.0.lock().unwrap().games();
        let mut timing = timing_logger.lock().unwrap();
        if let Err(error) = timing.finish(config.training.total_updates, games, positions, started.elapsed()) {
            eprintln!("failed to write run timing: {error}");
            process::exit(1);
        }
        println!("finished {} updates; self-play generated {games} games and {positions} positions ({replay_games} retained)",
            config.training.total_updates);
        process::exit(0)
    })
}
