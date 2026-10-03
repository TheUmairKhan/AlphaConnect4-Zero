use rand::RngExt;

use crate::board::{Board, GameResult};

const WIN: i32 = 1_000_000;
const INF: i32 = 2 * WIN;
const ORDER: [u8; 7] = [3, 2, 4, 1, 5, 0, 6];
const CENTER_COLUMN: u64 = 0x3F << 21;
const WINDOWS: [u64; 69] = build_windows();

const fn build_windows() -> [u64; 69] {
    let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let mut windows = [0u64; 69];
    let mut count = 0;
    let mut d = 0;
    while d < 4 {
        let (dc, dr) = directions[d];
        let mut column = 0;
        while column < 7 {
            let mut row = 0;
            while row < 6 {
                let end_column = column + 3 * dc;
                let end_row = row + 3 * dr;
                if end_column >= 0 && end_column < 7 && end_row >= 0 && end_row < 6 {
                    let mut mask = 0u64;
                    let mut i = 0;
                    while i < 4 {
                        let c = column + i * dc;
                        let r = row + i * dr;
                        mask |= 1u64 << (c * 7 + r);
                        i += 1;
                    }
                    windows[count] = mask;
                    count += 1;
                }
                row += 1;
            }
            column += 1;
        }
        d += 1;
    }
    windows
}

pub fn best_action(board: &Board, depth: u32) -> u8 {
    let depth = depth.max(1);
    let moves = board.valid_moves();
    let mut best_moves = Vec::new();
    let mut best_score = -INF;

    for &action in ORDER.iter().filter(|a| moves.contains(a)) {
        let score = score_move(board, depth, action, -INF, INF);
        if score > best_score {
            best_score = score;
            best_moves.clear();
            best_moves.push(action);
        } else if score == best_score {
            best_moves.push(action);
        }
    }

    best_moves[rand::rng().random_range(0..best_moves.len())]
}

fn score_move(board: &Board, depth: u32, action: u8, alpha: i32, beta: i32) -> i32 {
    let mut child = *board;
    match child.place_piece(action) {
        GameResult::Win(_) => WIN + depth as i32,
        GameResult::Draw => 0,
        GameResult::Ongoing => -negamax(&child, depth - 1, -beta, -alpha),
    }
}

fn negamax(board: &Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    if depth == 0 {
        return heuristic(board);
    }

    let moves = board.valid_moves();
    let mut best = -INF;

    for &action in ORDER.iter().filter(|a| moves.contains(a)) {
        let value = score_move(board, depth, action, alpha, beta);
        best = best.max(value);
        alpha = alpha.max(value);
        if alpha >= beta {
            break;
        }
    }

    best
}

fn heuristic(board: &Board) -> i32 {
    let me = board.pieces(board.current_player());
    let opponent = board.pieces(board.current_player().other());

    let mut score = 3 * ((me & CENTER_COLUMN).count_ones() as i32
        - (opponent & CENTER_COLUMN).count_ones() as i32);

    for window in WINDOWS {
        let mine = (window & me).count_ones();
        let theirs = (window & opponent).count_ones();
        score += match (mine, theirs) {
            (2, 0) => 2,
            (3, 0) => 5,
            (0, 2) => -2,
            (0, 3) => -5,
            _ => 0,
        };
    }

    score
}

#[cfg(test)]
mod tests {
    use super::{best_action, WINDOWS};
    use crate::board::Board;

    fn play(moves: &[u8]) -> Board {
        let mut board = Board::new();
        for &column in moves {
            board.place_piece(column);
        }
        board
    }

    #[test]
    fn windows_are_distinct_four_cell_masks() {
        for (i, window) in WINDOWS.iter().enumerate() {
            assert_eq!(window.count_ones(), 4);
            assert!(!WINDOWS[..i].contains(window));
        }
    }

    #[test]
    fn takes_an_immediate_win_over_blocking() {
        let board = play(&[0, 1, 0, 1, 0, 1]);
        for depth in 1..=6 {
            assert_eq!(best_action(&board, depth), 0);
        }
    }

    #[test]
    fn blocks_an_immediate_threat() {
        let board = play(&[0, 1, 0, 1, 0]);
        for depth in 2..=6 {
            assert_eq!(best_action(&board, depth), 0);
        }
    }

    #[test]
    fn finds_a_double_threat() {
        let board = play(&[2, 2, 3, 3]);
        for depth in 3..=6 {
            let action = best_action(&board, depth);
            assert!(action == 1 || action == 4, "depth {depth} chose {action}");
        }
    }
}
