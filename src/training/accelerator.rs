use std::sync::mpsc::Receiver;

use crate::{board::Board, mcts::evaluation::{EvalMessage, EvalOutput}};

#[derive(Default)]
pub struct BatchSummary {
    pub batches: usize,
    pub requests: usize,
    pub largest_batch: usize,
}

pub fn serve_requests<F>(
    request_receiver: Receiver<EvalMessage>,
    worker_count: usize,
    batch_size: usize,
    mut evaluate: F,
) -> BatchSummary
where
    F: FnMut(&[Board]) -> Vec<EvalOutput>,
{
    let mut summary = BatchSummary::default();
    let mut active_workers = worker_count;
    let mut requests = Vec::with_capacity(batch_size);

    while active_workers > 0 {
        match request_receiver.recv().expect("evaluation workers stopped unexpectedly") {
            EvalMessage::Request(request) => requests.push(request),
            EvalMessage::WorkerFinished => active_workers -= 1,
        }

        if requests.len() == batch_size || (!requests.is_empty() && requests.len() == active_workers) {
            let boards: Vec<Board> = requests.iter().map(|request| request.board).collect();
            let outputs = evaluate(&boards);
            summary.batches += 1;
            summary.requests += requests.len();
            summary.largest_batch = summary.largest_batch.max(requests.len());

            for (request, output) in requests.drain(..).zip(outputs) {
                request.sender.send(output).expect("evaluation client stopped before receiving its result");
            }
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use crate::{board::Board, mcts::evaluation::{EvalMessage, EvalOutput, LeafRequest}};

    use super::serve_requests;

    #[test]
    fn full_batches_route_results() {
        let (request_sender, request_receiver) = mpsc::channel();
        let mut replies = Vec::new();
        for pieces in 0..4 {
            let mut board = Board::new();
            for _ in 0..pieces {
                board.place_piece(0);
            }
            let (sender, receiver) = mpsc::sync_channel(1);
            request_sender.send(EvalMessage::Request(LeafRequest { board, sender })).unwrap();
            replies.push(receiver);
        }
        for _ in 0..4 {
            request_sender.send(EvalMessage::WorkerFinished).unwrap();
        }
        drop(request_sender);

        let mut batch_sizes = Vec::new();
        let summary = serve_requests(request_receiver, 4, 2, |boards| {
            batch_sizes.push(boards.len());
            boards.iter().map(|board| EvalOutput {
                priors: [1.0 / 7.0; 7],
                value: board.encode_state().iter().flatten().flatten().sum(),
            }).collect()
        });

        assert_eq!(batch_sizes, [2, 2]);
        assert_eq!(summary.requests, 4);
        assert_eq!(summary.batches, 2);
        assert_eq!(summary.largest_batch, 2);
        for (pieces, receiver) in replies.into_iter().enumerate() {
            assert_eq!(receiver.recv().unwrap().value, pieces as f32);
        }
    }

    #[test]
    fn waits_for_a_full_batch() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (first_sender, first_reply) = mpsc::sync_channel(1);
        let (second_sender, second_reply) = mpsc::sync_channel(1);

        thread::scope(|scope| {
            let service = scope.spawn(move || {
                serve_requests(request_receiver, 2, 2, |boards| {
                    boards.iter().map(|_| EvalOutput { priors: [1.0 / 7.0; 7], value: 0.5 }).collect()
                })
            });
            request_sender.send(EvalMessage::Request(LeafRequest { board: Board::new(), sender: first_sender })).unwrap();
            let replied_early = first_reply.recv_timeout(Duration::from_millis(20)).is_ok();
            request_sender.send(EvalMessage::Request(LeafRequest { board: Board::new(), sender: second_sender })).unwrap();
            assert_eq!(first_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            assert_eq!(second_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            for _ in 0..2 {
                request_sender.send(EvalMessage::WorkerFinished).unwrap();
            }
            drop(request_sender);
            assert_eq!(service.join().unwrap().requests, 2);
            assert!(!replied_early);
        });
    }

    #[test]
    fn flushes_remaining_requests_when_fewer_workers_remain_than_batch_size() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (first_sender, first_reply) = mpsc::sync_channel(1);
        let (second_sender, second_reply) = mpsc::sync_channel(1);

        thread::scope(|scope| {
            let service = scope.spawn(move || {
                serve_requests(request_receiver, 3, 3, |boards| {
                    boards.iter().map(|_| EvalOutput { priors: [1.0 / 7.0; 7], value: 0.5 }).collect()
                })
            });
            request_sender.send(EvalMessage::Request(LeafRequest { board: Board::new(), sender: first_sender })).unwrap();
            request_sender.send(EvalMessage::Request(LeafRequest { board: Board::new(), sender: second_sender })).unwrap();
            request_sender.send(EvalMessage::WorkerFinished).unwrap();
            assert_eq!(first_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            assert_eq!(second_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            for _ in 0..2 {
                request_sender.send(EvalMessage::WorkerFinished).unwrap();
            }
            drop(request_sender);
            let summary = service.join().unwrap();
            assert_eq!(summary.requests, 2);
            assert_eq!(summary.largest_batch, 2);
        });
    }
}
