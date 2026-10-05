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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{FILTERED_PIECES, canonical_key, candidates, forced_outcome, play, suite, undecided};

    #[test]
    fn suite_starts_with_the_empty_board_and_center_columns() {
        let openings = suite(50);
        assert_eq!(openings.len(), 50);
        assert_eq!(openings[..4], [vec![], vec![2], vec![3], vec![4]]);
        assert!(openings[4..].iter().all(|moves| (2..=4).contains(&moves.len())));
    }

    #[test]
    fn sampled_openings_are_distinct_up_to_mirroring_and_undecided() {
        let openings = suite(50);
        let keys: HashSet<_> = openings[4..].iter().map(|moves| canonical_key(&play(moves))).collect();
        assert_eq!(keys.len(), 46);
        assert!(openings[4..].iter().all(|moves| moves.len() < FILTERED_PIECES || undecided(moves)));
    }

    #[test]
    fn buckets_are_drawn_evenly_until_one_runs_out() {
        let openings = suite(4 + 60);
        let count = |pieces: usize| openings.iter().filter(|moves| moves.len() == pieces).count();
        assert!((2..=4).all(|pieces| (12..=28).contains(&count(pieces))), "{:?}", (count(2), count(3), count(4)));

        let two_piece = candidates(2).len();
        let openings = suite(4 + 2 * two_piece + 20);
        assert_eq!(openings.iter().filter(|moves| moves.len() == 2).count(), two_piece);
    }

    #[test]
    fn two_and_three_piece_openings_never_allow_a_forced_win() {
        assert!(candidates(2).iter().chain(&candidates(3)).all(|moves| undecided(moves)));
    }

    #[test]
    fn suite_is_the_same_every_time() {
        assert_eq!(suite(50), suite(50));
    }

    #[test]
    fn small_suites_keep_the_empty_board_first() {
        assert_eq!(suite(2), [vec![], vec![2]]);
    }

    #[test]
    fn open_two_on_the_bottom_row_is_a_forced_win_by_ply_seven() {
        let moves = [2, 0, 3, 0];
        assert_eq!(forced_outcome(&play(&moves), 3), 1);
        assert!(!undecided(&moves));
    }

    #[test]
    fn stacked_center_is_undecided_through_ply_nine() {
        assert!(undecided(&[3, 3]));
    }

    #[test]
    fn mirrored_positions_share_a_key() {
        assert_eq!(canonical_key(&play(&[0, 1, 2])), canonical_key(&play(&[6, 5, 4])));
        assert_ne!(canonical_key(&play(&[0, 1, 2])), canonical_key(&play(&[0, 1, 3])));
    }
}
