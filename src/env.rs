use crate::board::{Board, GameResult};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Connect4Env {
    board: Board
}

impl Connect4Env {
    pub fn new() -> Self {
        Self { board: Board::new() }
    }

    pub fn reset(&mut self) {
        self.board = Board::new()
    }

    pub fn state(&self) -> &Board {
        &self.board
    }

    pub fn step(&mut self, action: u8) -> GameResult{
        self.board.place_piece(action);
        self.board.result()
    }
}