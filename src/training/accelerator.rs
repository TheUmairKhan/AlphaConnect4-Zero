use std::sync::mpsc::Receiver;

use crate::{board::Board, mcts::evaluation::{EvalOutput, LeafRequest}};

pub fn serve_requests<F>(
    request_receiver: Receiver<LeafRequest>,
    batch_size: usize,
    mut evaluate: F,
)
where
    F: FnMut(&[Board]) -> Vec<EvalOutput>,
{
    let mut requests = Vec::with_capacity(batch_size);

    for request in request_receiver {
        requests.push(request);

        if requests.len() == batch_size {
            let boards: Vec<Board> = requests.iter().map(|request| request.board).collect();
            let outputs = evaluate(&boards);

            for (request, output) in requests.drain(..).zip(outputs) {
                request.sender.send(output).expect("evaluation client stopped before receiving its result");
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use crate::{board::Board, mcts::evaluation::{EvalOutput, LeafRequest}};

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
            request_sender.send(LeafRequest { board, sender }).unwrap();
            replies.push(receiver);
        }
        drop(request_sender);

        let mut batch_sizes = Vec::new();
        serve_requests(request_receiver, 2, |boards| {
            batch_sizes.push(boards.len());
            boards.iter().map(|board| EvalOutput {
                priors: [1.0 / 7.0; 7],
                value: board.encode_state().iter().flatten().flatten().sum(),
            }).collect()
        });

        assert_eq!(batch_sizes, [2, 2]);
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
                serve_requests(request_receiver, 2, |boards| {
                    boards.iter().map(|_| EvalOutput { priors: [1.0 / 7.0; 7], value: 0.5 }).collect()
                })
            });
            request_sender.send(LeafRequest { board: Board::new(), sender: first_sender }).unwrap();
            let replied_early = first_reply.recv_timeout(Duration::from_millis(20)).is_ok();
            request_sender.send(LeafRequest { board: Board::new(), sender: second_sender }).unwrap();
            assert_eq!(first_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            assert_eq!(second_reply.recv_timeout(Duration::from_secs(1)).unwrap().value, 0.5);
            drop(request_sender);
            service.join().unwrap();
            assert!(!replied_early);
        });
    }
}
