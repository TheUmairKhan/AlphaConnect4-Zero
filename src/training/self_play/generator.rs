use std::{
    cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
    }, thread,
};

use crate::{
    config::TrainingConfig,
    mcts::evaluation::{BatchingClient, WorkerMessage},
    training::accelerator::worker::{LeafQueue, drive_games},
};

use super::self_play::{SelfPlayGame, self_play};

/// Runs self-play workers until the training process exits.
/// Each completed game is passed to `on_game` as soon as a worker finishes it.
pub fn generate_self_play_games<F, G>(
    config: &TrainingConfig,
    serve_requests: F,
    mut on_game: G,
)
where
    F: FnOnce(Receiver<WorkerMessage>),
    G: FnMut(usize, SelfPlayGame) + Send,
{
    let worker_count = config.self_play.threads;
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
                let queue = Rc::new(RefCell::new(Vec::new()));
                let games = (0..config.self_play.games_per_thread)
                    .map(|_| {
                        Box::pin(play_games(queue.clone(), config, next_game, completed_sender.clone()))
                            as Pin<Box<dyn Future<Output = ()>>>
                    })
                    .collect();
                drive_games(games, &queue, &request_sender);
            });
        }
        drop(request_sender);
        drop(completed_sender);
        serve_requests(request_receiver);
    })
}

async fn play_games(
    queue: Rc<LeafQueue>,
    config: &TrainingConfig,
    next_game: &AtomicUsize,
    completed_sender: SyncSender<(usize, SelfPlayGame)>,
) {
    loop {
        let client = BatchingClient::new(queue.clone());
        let game = self_play(client, config).await.expect("self-play game failed");
        let n = next_game.fetch_add(1, Ordering::Relaxed);
        completed_sender.send((n + 1, game)).expect("completed-game receiver closed");
    }
}
