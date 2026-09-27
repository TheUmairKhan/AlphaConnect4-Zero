use std::{error::Error, fs, path::Path};

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
    println!("Live metrics: {}", checkpoint_dir.join("metrics.html").display());
    best.clone().save_file(checkpoint_dir.join("best-step-0"), &DefaultRecorder::new())?;

    for iteration in 1..=config.training.iterations {
        let mut positions = 0;
        for _ in 0..config.training.games_per_iteration {
            let examples = self_play(
                ZeroNetPolicy::new(best.clone(), device.clone()),
                config,
            )?;
            positions += examples.len();
            replay.push(examples);
        }

        if replay.positions() < config.training.batch_size {
            return Err("not enough self-play positions for one training batch".into());
        }

        let mut last_loss = None;
        for _ in 0..config.training.updates_per_iteration {
            let (states, policies, values) = replay.sample_batch::<B>(
                config.training.batch_size,
                device,
                &mut rng,
            );
            let (logits, predictions) = model.forward(states);
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
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.training.learning_rate, model, gradients);
            training_steps += 1;
            metrics_logger.record(metrics)?;

            if training_steps % config.evaluation.every_training_steps == 0 {
                let candidate = model.valid();
                candidate.clone().save_file(
                    checkpoint_dir.join(format!("candidate-step-{training_steps}")),
                    &DefaultRecorder::new(),
                )?;
                let score = evaluate(
                    &candidate,
                    &best,
                    config,
                    device,
                );
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
    }

    Ok(())
}
