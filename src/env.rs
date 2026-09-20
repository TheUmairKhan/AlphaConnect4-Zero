use crate::board::{Board, GameResult};

struct Connect4Env {
    board: Board
}

impl Connect4Env {
    fn new() -> Self {
        Self { board: Board::new() }
    }

    fn reset(&mut self) {
        self.board = Board::new()
    }

    fn state(&self) -> &Board {
        &self.board
    }

    fn step(&mut self, action: u8) -> GameResult{
        self.board.place_piece(action);
        self.board.result()
    }
}