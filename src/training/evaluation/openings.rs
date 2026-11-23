use std::collections::HashSet;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::board::{Board, GameResult, Player};

const FIXED_OPENINGS: [&[u8]; 4] = [&[], &[2], &[3], &[4]];
const SAMPLED_PIECES: [u32; 3] = [2, 3, 4];
const FILTERED_PIECES: usize = 4;
const UNDECIDED_THROUGH_PLY: u32 = 9;
const SEED: u64 = 4;

pub fn suite(count: usize) -> Vec<Vec<u8>> {
    let mut openings: Vec<Vec<u8>> = FIXED_OPENINGS.iter().take(count).map(|moves| moves.to_vec()).collect();
    let mut buckets: Vec<Vec<Vec<u8>>> = SAMPLED_PIECES.iter().map(|&pieces| candidates(pieces)).collect();
    let mut rng = StdRng::seed_from_u64(SEED);

    while openings.len() < count {
        let remaining: Vec<usize> = (0..buckets.len()).filter(|&b| !buckets[b].is_empty()).collect();
        let bucket = &mut buckets[remaining[rng.random_range(0..remaining.len())]];
        let moves = bucket.swap_remove(rng.random_range(0..bucket.len()));
        if moves.len() < FILTERED_PIECES || undecided(&moves) {
            openings.push(moves);
        }
    }

    openings
}

fn play(moves: &[u8]) -> Board {
    let mut board = Board::new();
    for &column in moves {
        board.place_piece(column);
    }
    board
}

fn candidates(pieces: u32) -> Vec<Vec<u8>> {
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for code in 0..7usize.pow(pieces) {
        let moves: Vec<u8> = (0..pieces).map(|i| (code / 7usize.pow(i) % 7) as u8).collect();
        if seen.insert(canonical_key(&play(&moves))) {
            candidates.push(moves);
        }
    }
    candidates
}

fn undecided(moves: &[u8]) -> bool {
    forced_outcome(&play(moves), UNDECIDED_THROUGH_PLY - moves.len() as u32) == 0
}

fn forced_outcome(board: &Board, plies: u32) -> i8 {
    if plies == 0 {
        return 0;
    }
    let mut best = -1;
    for column in board.valid_moves() {
        let mut child = *board;
        let value = match child.place_piece(column) {
            GameResult::Win(_) => 1,
            GameResult::Draw => 0,
            GameResult::Ongoing => -forced_outcome(&child, plies - 1),
        };
        best = best.max(value);
        if best == 1 {
            break;
        }
    }
    best
}

fn canonical_key(board: &Board) -> (u64, u64) {
    let key = (board.pieces(Player::Red), board.pieces(Player::Yellow));
    key.min((mirror(key.0), mirror(key.1)))
}

fn mirror(bits: u64) -> u64 {
    (0..7).fold(0, |mirrored, column| mirrored | ((bits >> (7 * column)) & 0x3F) << (7 * (6 - column)))
}
