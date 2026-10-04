use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::mpsc::{self, SyncSender},
    task::{Context, Waker},
};

use futures::channel::oneshot;

use crate::{board::Board, mcts::evaluation::{BatchRequest, EvalOutput, WorkerMessage}};

pub type LeafQueue = RefCell<Vec<(Board, oneshot::Sender<EvalOutput>)>>;

pub fn drive_games<'a>(
    mut games: Vec<Pin<Box<dyn Future<Output = ()> + 'a>>>,
    queue: &LeafQueue,
    request_sender: &SyncSender<WorkerMessage>,
) {
    let mut context = Context::from_waker(Waker::noop());
    let (reply_sender, reply_receiver) = mpsc::sync_channel(1);

    loop {
        games.retain_mut(|game| game.as_mut().poll(&mut context).is_pending());
        if games.is_empty() {
            request_sender.send(WorkerMessage::Finished).expect("evaluation request queue closed");
            return;
        }

        let (boards, senders): (Vec<_>, Vec<_>) = queue.borrow_mut().drain(..).unzip();
        request_sender.send(WorkerMessage::Batch(BatchRequest { boards, sender: reply_sender.clone() }))
            .expect("evaluation request queue closed");
        let outputs = reply_receiver.recv().expect("evaluation reply channel closed");
        for (sender, output) in senders.into_iter().zip(outputs) {
            let _ = sender.send(output);
        }
    }
}
