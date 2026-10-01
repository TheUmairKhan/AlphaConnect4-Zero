use std::{
    cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
    }, task::{Context, Waker}, thread,
};

use futures::channel::oneshot;

use crate::{board::Board, mcts::evaluation::{BatchRequest, BatchingClient, EvalOutput}};

use super::{config::TrainingConfig, self_play::{SelfPlayGame, self_play}};

/// Runs self-play workers until the training process exits.
/// Each completed game is passed to `on_game` as soon as a worker finishes it.
pub fn generate_self_play_games<F, G>(
    config: &TrainingConfig,
    serve_requests: F,
    mut on_game: G,
)
where
    F: FnOnce(Receiver<BatchRequest>),
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
                let mut games: Vec<Pin<Box<dyn Future<Output = ()> + '_>>> = (0..config.self_play.games_per_thread)
                    .map(|_| {
                        let queue = queue.clone();
                        let completed_sender = completed_sender.clone();
                        Box::pin(play_games(queue, config, next_game, completed_sender))
                            as Pin<Box<dyn Future<Output = ()>>>
                    })
                    .collect();

                let mut context = Context::from_waker(Waker::noop());
                let (reply_sender, reply_receiver) = mpsc::sync_channel(1);
                loop {
                    for game in &mut games {
                        let _ = game.as_mut().poll(&mut context);
                    }

                    let (boards, senders): (Vec<_>, Vec<_>) = queue.borrow_mut().drain(..).unzip();
                    request_sender.send(BatchRequest { boards, sender: reply_sender.clone() })
                        .expect("evaluation request queue closed");
                    let outputs = reply_receiver.recv().expect("evaluation reply channel closed");
                    for (sender, output) in senders.into_iter().zip(outputs) {
                        let _ = sender.send(output);
                    }
                }
            });
        }
        drop(request_sender);
        drop(completed_sender);
        serve_requests(request_receiver);
    })
}

async fn play_games(
    queue: Rc<RefCell<Vec<(Board, oneshot::Sender<EvalOutput>)>>>,
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
