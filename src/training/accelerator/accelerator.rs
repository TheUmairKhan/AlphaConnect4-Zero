use std::sync::mpsc::Receiver;

use crate::{board::Board, mcts::evaluation::{BatchRequest, EvalOutput, WorkerMessage}};

pub fn serve_requests<F>(
    request_receiver: Receiver<WorkerMessage>,
    threads: usize,
    mut evaluate: F,
)
where
    F: FnMut(&[Board]) -> Vec<EvalOutput>,
{
    let mut requests: Vec<BatchRequest> = Vec::with_capacity(threads);
    let mut boards = Vec::new();
    let mut active = threads;

    for message in request_receiver {
        match message {
            WorkerMessage::Batch(request) => requests.push(request),
            WorkerMessage::Finished => active -= 1,
        }

        if !requests.is_empty() && requests.len() == active {
            boards.clear();
            boards.extend(requests.iter().flat_map(|request| request.boards.iter().copied()));
            let mut outputs = if boards.is_empty() { Vec::new() } else { evaluate(&boards) }.into_iter();

            for request in requests.drain(..) {
                let reply = outputs.by_ref().take(request.boards.len()).collect();
                request.sender.send(reply).expect("worker thread stopped before receiving its results");
            }
        }

        if active == 0 {
            break;
        }
    }
}
