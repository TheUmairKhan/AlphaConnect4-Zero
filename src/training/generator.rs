use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
};

use crate::mcts::evaluation::{EvalClient, EvalMessage};

use super::{config::TrainingConfig, self_play::{SelfPlayGame, self_play}};

/// Runs self-play workers while `serve_requests` handles their leaf evaluations.
/// Each completed game is passed to `on_game` as soon as a worker finishes it.
pub fn generate_self_play_games<F, G, R>(
    config: &TrainingConfig,
    serve_requests: F,
    mut on_game: G,
) -> R
where
    F: FnOnce(Receiver<EvalMessage>, usize) -> R,
    G: FnMut(usize, SelfPlayGame) + Send,
{
    let game_count = config.training.games_per_iteration;
    let worker_count = config.self_play.parallel_games.min(game_count);
    let (request_sender, request_receiver) = mpsc::sync_channel(worker_count);
    let (completed_sender, completed_receiver) = mpsc::sync_channel(worker_count);
    let next_game = AtomicUsize::new(0);

    thread::scope(|scope| {
        let collector = scope.spawn(move || {
            for (game_number, game) in completed_receiver {
                on_game(game_number, game);
            }
        });

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let request_sender = request_sender.clone();
            let completed_sender = completed_sender.clone();
            let next_game = &next_game;
            workers.push(scope.spawn(move || {
                loop {
                    let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                    if game_index >= game_count {
                        break;
                    }
                    let client = EvalClient::new(request_sender.clone());
                    let game = self_play(client, config).expect("self-play game failed");
                    completed_sender.send((game_index + 1, game))
                        .expect("completed-game receiver closed");
                }
                request_sender.send(EvalMessage::WorkerFinished)
                    .expect("evaluation request queue closed");
            }));
        }
        drop(request_sender);
        drop(completed_sender);
        let service_result = serve_requests(request_receiver, worker_count);

        for worker in workers {
            worker.join().expect("self-play worker panicked");
        }
        collector.join().expect("completed-game collector panicked");
        service_result
    })
}

#[cfg(test)]
mod tests {
    use crate::mcts::evaluation::{EvalMessage, EvalOutput};

    use super::{super::config::TrainingConfig, generate_self_play_games};

    #[test]
    fn generates_all_games_with_a_request_serving_loop() {
        let mut config = TrainingConfig::load(concat!(env!("CARGO_MANIFEST_DIR"), "/training.toml")).unwrap();
        config.training.games_per_iteration = 3;
        config.self_play.parallel_games = 2;
        config.self_play.simulations = 1;

        let mut completed = Vec::new();
        generate_self_play_games(
            &config,
            |request_receiver, _worker_count| {
                for message in request_receiver {
                    if let EvalMessage::Request(request) = message {
                        request.sender.send(EvalOutput {
                            priors: [1.0 / 7.0; 7],
                            value: 0.0,
                        }).unwrap();
                    }
                }
            },
            |game_number, game| {
                completed.push(game_number);
                assert!(!game.examples.is_empty());
            },
        );

        completed.sort_unstable();
        assert_eq!(completed, [1, 2, 3]);
    }
}
