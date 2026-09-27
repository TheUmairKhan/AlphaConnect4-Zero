use std::{error::Error, fs, path::Path, time::{Duration, Instant}};

use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::DefaultRecorder,
    tensor::{activation::log_softmax, backend::AutodiffBackend},
};
use crate::{
    mcts::policy::ZeroNetPolicy,
    model::zeronet::{ZeroNet, ZeroNetConfig},
    training::{
        config::TrainingConfig,
        evaluation::evaluate,
        l2::squared_l2,
        metrics::{MetricsLogger, StepMetrics},
        performance::{TimingLogger, UpdateTiming},
        replay::ReplayBuffer,
        self_play::self_play,
    },
};

pub fn run<B: AutodiffBackend>(config: &TrainingConfig, device: &B::Device) -> Result<(), Box<dyn Error>> {
    let mut model = ZeroNetConfig::new(2, config.model.hidden_size, config.model.num_res_blocks).init::<B>(device);
    let mut best = model.valid();
    let mut optimizer = AdamConfig::new().init::<B, ZeroNet<B>>();
    let mut replay = ReplayBuffer::new(config.training.replay_capacity_games);
    let mut rng = rand::rng();
    let mut training_steps = 0;
    let checkpoint_dir = Path::new(&config.training.checkpoint_dir);
    fs::create_dir_all(checkpoint_dir)?;
    let mut metrics_logger = MetricsLogger::new(checkpoint_dir)?;
    let mut timing_logger = TimingLogger::new(checkpoint_dir)?;
    println!("Live metrics: {}", checkpoint_dir.join("metrics.html").display());
    println!("Performance: {}", checkpoint_dir.join("performance.html").display());
    best.clone().save_file(checkpoint_dir.join("best-step-0"), &DefaultRecorder::new())?;

    for iteration in 1..=config.training.iterations {
        let iteration_started = Instant::now();
        let self_play_started = Instant::now();
        let mut positions = 0;
        for game_number in 1..=config.training.games_per_iteration {
            let game = self_play(
                ZeroNetPolicy::new(best.clone(), device.clone()),
                config,
            )?;
            positions += game.examples.len();
            timing_logger.game(iteration, game_number, &game.timing, config.self_play.simulations)?;
            replay.push(game.examples);
        }
        let self_play_elapsed = self_play_started.elapsed();

        if replay.positions() < config.training.batch_size {
            return Err("not enough self-play positions for one training batch".into());
        }

        let mut last_loss = None;
        let mut updates_elapsed = Duration::ZERO;
        let mut evaluation_elapsed = Duration::ZERO;
        for _ in 0..config.training.updates_per_iteration {
            let update_started = Instant::now();
            let (states, policies, values) = replay.sample_batch::<B>(
                config.training.batch_size,
                device,
                &mut rng,
            );
            let sample_elapsed = update_started.elapsed();
            let forward_started = Instant::now();
            let (logits, predictions) = model.forward(states);
            let forward_elapsed = forward_started.elapsed();
            let loss_started = Instant::now();
            let log_policy = log_softmax(logits, 1);
            let policy_loss = -(policies.clone() * log_policy.clone()).sum_dim(1).mean();
            let error = predictions - values;
            let value_loss = (error.clone() * error.clone()).mean();
            let metrics = StepMetrics::from_batch(
                training_steps + 1,
                &policy_loss,
                &value_loss,
                log_policy,
                policies,
                error,
            );
            let loss = policy_loss + value_loss
                + squared_l2(&model, device) * config.training.l2_coefficient;
            last_loss = Some(loss.clone().into_scalar());
            let loss_elapsed = loss_started.elapsed();
            let backward_started = Instant::now();
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            let backward_elapsed = backward_started.elapsed();
            let optimizer_started = Instant::now();
            model = optimizer.step(config.training.learning_rate, model, gradients);
            let optimizer_elapsed = optimizer_started.elapsed();
            training_steps += 1;
            let logging_started = Instant::now();
            metrics_logger.record(metrics)?;
            let logging_elapsed = logging_started.elapsed();
            let update_total = update_started.elapsed();
            updates_elapsed += update_total;
            timing_logger.update(training_steps, UpdateTiming {
                total: update_total,
                sample: sample_elapsed,
                forward: forward_elapsed,
                loss_metrics: loss_elapsed,
                backward: backward_elapsed,
                optimizer: optimizer_elapsed,
                logging: logging_elapsed,
            })?;

            if training_steps % config.evaluation.every_training_steps == 0 {
                let candidate = model.valid();
                candidate.clone().save_file(
                    checkpoint_dir.join(format!("candidate-step-{training_steps}")),
                    &DefaultRecorder::new(),
                )?;
                let evaluation_started = Instant::now();
                let score = evaluate(
                    &candidate,
                    &best,
                    config,
                    device,
                );
                let elapsed = evaluation_started.elapsed();
                let win_rate = score.candidate_wins as f32 / config.evaluation.games as f32;
                println!(
                    "evaluation at step {training_steps}: candidate {} wins, best {} wins, {} draws ({:.1}% candidate win rate)",
                    score.candidate_wins,
                    score.best_wins,
                    score.draws,
                    win_rate * 100.0,
                );
                if win_rate > config.evaluation.promotion_win_rate {
                    best = candidate;
                    best.clone().save_file(
                        checkpoint_dir.join(format!("best-step-{training_steps}")),
                        &DefaultRecorder::new(),
                    )?;
                    println!("promoted candidate at step {training_steps}");
                }
                evaluation_elapsed += elapsed;
                timing_logger.evaluation(training_steps, config.evaluation.games, elapsed)?;
            }
        }

        println!("iteration {iteration}/{}: {} games, {positions} positions, replay {} games/{} positions, step {training_steps}, loss {last_loss:?}",
            config.training.iterations, config.training.games_per_iteration, replay.games(), replay.positions());

        if iteration % config.training.checkpoint_every_iterations == 0
            || iteration == config.training.iterations
        {
            let path = checkpoint_dir.join(format!("model-iteration-{iteration}"));
            model.valid().save_file(path, &DefaultRecorder::new())?;
        }
        timing_logger.iteration(iteration, config.training.games_per_iteration, positions,
            iteration_started.elapsed(), self_play_elapsed, updates_elapsed, evaluation_elapsed)?;
    }

    Ok(())
}
