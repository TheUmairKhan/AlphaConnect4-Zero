use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
};

use crate::mcts::evaluation::{EvalClient, LeafRequest};

use super::{config::TrainingConfig, self_play::{SelfPlayGame, self_play}};

/// Runs self-play workers until the training process exits.
/// Each completed game is passed to `on_game` as soon as a worker finishes it.
pub fn generate_self_play_games<F, G>(
    config: &TrainingConfig,
    serve_requests: F,
    mut on_game: G,
)
where
    F: FnOnce(Receiver<LeafRequest>),
    G: FnMut(usize, SelfPlayGame) + Send,
{
    let worker_count = config.self_play.parallel_games;
    let (request_sender, request_receiver) = mpsc::sync_channel(worker_count);
    let (completed_sender, completed_receiver) = mpsc::sync_channel(worker_count);
    let next_game = AtomicUsize::new(0);

    thread::scope(|scope| {
        scope.spawn(move || {
            for (game_number, game) in completed_receiver {
                on_game(game_number, game);
            }
        });

        for _ in 0..worker_count {
            let request_sender = request_sender.clone();
            let completed_sender = completed_sender.clone();
            let next_game = &next_game;
            scope.spawn(move || {
                loop {
                    let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                    let client = EvalClient::new(request_sender.clone());
                    let game = self_play(client, config).expect("self-play game failed");
                    completed_sender.send((game_index + 1, game))
                        .expect("completed-game receiver closed");
                }
            });
        }
        drop(request_sender);
        drop(completed_sender);
        serve_requests(request_receiver);
    })
}
