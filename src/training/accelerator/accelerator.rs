use std::sync::mpsc::Receiver;

use crate::{board::Board, mcts::evaluation::{BatchRequest, EvalOutput}};

pub fn serve_requests<F>(
    request_receiver: Receiver<BatchRequest>,
    threads: usize,
    mut evaluate: F,
)
where
    F: FnMut(&[Board]) -> Vec<EvalOutput>,
{
    let mut requests = Vec::with_capacity(threads);
    let mut boards = Vec::new();

    for request in request_receiver {
        requests.push(request);

        if requests.len() == threads {
            boards.clear();
            boards.extend(requests.iter().flat_map(|request| request.boards.iter().copied()));
            let mut outputs = evaluate(&boards).into_iter();

            for request in requests.drain(..) {
                let reply = outputs.by_ref().take(request.boards.len()).collect();
                request.sender.send(reply).expect("self-play thread stopped before receiving its results");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use crate::{board::Board, mcts::evaluation::{BatchRequest, EvalOutput}};

    use super::serve_requests;

    fn board_with(pieces: usize) -> Board {
        let mut board = Board::new();
        for _ in 0..pieces {
            board.place_piece(0);
        }
        board
    }

    #[test]
    fn joins_thread_batches_and_routes_results() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (first_sender, first_reply) = mpsc::sync_channel(1);
        let (second_sender, second_reply) = mpsc::sync_channel(1);
        request_sender.send(BatchRequest { boards: vec![board_with(0), board_with(1)], sender: first_sender }).unwrap();
        request_sender.send(BatchRequest { boards: vec![board_with(2), board_with(3), board_with(4)], sender: second_sender }).unwrap();
        drop(request_sender);

        let mut batch_sizes = Vec::new();
        serve_requests(request_receiver, 2, |boards| {
            batch_sizes.push(boards.len());
            boards.iter().map(|board| EvalOutput {
                priors: [1.0 / 7.0; 7],
                value: board.encode_state().iter().flatten().flatten().sum(),
            }).collect()
        });

        assert_eq!(batch_sizes, [5]);
        let values = |reply: mpsc::Receiver<Vec<EvalOutput>>| {
            reply.recv().unwrap().iter().map(|output| output.value).collect::<Vec<_>>()
        };
        assert_eq!(values(first_reply), [0.0, 1.0]);
        assert_eq!(values(second_reply), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn waits_for_every_thread() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (first_sender, first_reply) = mpsc::sync_channel(1);
        let (second_sender, second_reply) = mpsc::sync_channel(1);

        thread::scope(|scope| {
            let service = scope.spawn(move || {
                serve_requests(request_receiver, 2, |boards| {
                    boards.iter().map(|_| EvalOutput { priors: [1.0 / 7.0; 7], value: 0.5 }).collect()
                })
            });
            request_sender.send(BatchRequest { boards: vec![Board::new()], sender: first_sender }).unwrap();
            let replied_early = first_reply.recv_timeout(Duration::from_millis(20)).is_ok();
            request_sender.send(BatchRequest { boards: vec![Board::new()], sender: second_sender }).unwrap();
            assert_eq!(first_reply.recv_timeout(Duration::from_secs(1)).unwrap()[0].value, 0.5);
            assert_eq!(second_reply.recv_timeout(Duration::from_secs(1)).unwrap()[0].value, 0.5);
            drop(request_sender);
            service.join().unwrap();
            assert!(!replied_early);
        });
    }
}
