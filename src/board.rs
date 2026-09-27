#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    red: BitBoard,
    yellow: BitBoard,
    current_player: Player,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Red,
    Yellow,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]

pub enum GameResult {
    Ongoing,
    Win(Player),
    Draw,
}

impl Board {
    const BOTTOM_MASK : u64 = 
    (1 << 0) |
    (1 << 7) |
    (1 << 14)|
    (1 << 21)|
    (1 << 28)|
    (1 << 35)|
    (1 << 42);

    const BOARD_MASK : u64 =
    (0x3F << 0) |
    (0x3F << 7) |
    (0x3F << 14)|
    (0x3F << 21)|
    (0x3F << 28)|
    (0x3F << 35)|
    (0x3F << 42);

    pub fn new() -> Self {
        Self{
            red: BitBoard(0),
            yellow: BitBoard(0),
            current_player: Player::Red,
        }
    }

    fn occupied(&self) -> u64 {
        self.red.0 | self.yellow.0
    }

    pub fn valid_moves(&self) -> Vec<u8> {
        let occupied = self.occupied();
        let mut moves = Vec::new();
        
        for column in 0..7 {
            let top = 1u64 << (column * 7 + 5);
            if occupied & top == 0 {
                moves.push(column as u8);
            }
        }
        moves
    }

    pub fn place_piece(&mut self, column: u8) -> GameResult {
        let player = self.current_player;

        let shift = column as u64 * 7;
        let bottom = 1u64 << shift;
        let column_mask = 0x3Fu64 << shift;
        let pos = (self.occupied() + bottom) & column_mask;
        
        debug_assert!(pos != 0, "attempted move in full column");

        match player {
            Player::Red => self.red.0 |= pos,
            Player::Yellow => self.yellow.0 |= pos,
        }

        let result = self.result();

        self.current_player = player.other();

        result
    }

    fn has_won(&self, player: Player) -> bool {
        let board = match player {
            Player::Red => self.red.0,
            Player::Yellow => self.yellow.0,
        };

        // Vertical
        if board
            & (board >> 1)
            & (board >> 2)
            & (board >> 3)
            != 0
        {
            return true;
        }

        // Horizontal
        if board
            & (board >> 7)
            & (board >> 14)
            & (board >> 21)
            != 0
        {
            return true;
        }

        // Diagonal /
        if board
            & (board >> 6)
            & (board >> 12)
            & (board >> 18)
            != 0
        {
            return true;
        }

        // Diagonal \
        if board
            & (board >> 8)
            & (board >> 16)
            & (board >> 24)
            != 0
        {
            return true;
        }

        false
    }

    fn is_draw(&self) -> bool {
        (self.occupied() + Self::BOTTOM_MASK) & Self::BOARD_MASK == 0
    }

    pub fn result(&self) -> GameResult {
        if self.has_won(Player::Red) {
            GameResult::Win(Player::Red)
        } else if self.has_won(Player::Yellow) {
            GameResult::Win(Player::Yellow)
        } else if self.is_draw() {
            GameResult::Draw
        } else {
            GameResult::Ongoing
        }
    }

    pub fn current_player(&self) -> Player {
        self.current_player
    }

    pub fn encode_state(&self) -> [[[f32; 7]; 6]; 2] {
        let (current, opponent) = match self.current_player {
            Player::Red => (self.red.0, self.yellow.0),
            Player::Yellow => (self.yellow.0, self.red.0),
        };
        let mut planes = [[[0.0; 7]; 6]; 2];

        for column in 0..7 {
            for row in 0..6 {
                let bit = 1_u64 << (column * 7 + row);
                planes[0][row][column] = ((current & bit) != 0) as u8 as f32;
                planes[1][row][column] = ((opponent & bit) != 0) as u8 as f32;
            }
        }

        planes
    }
}


impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Board, GameResult, Player};

    fn play(moves: &[u8]) -> Board {
        let mut board = Board::new();
        for (index, &column) in moves.iter().enumerate() {
            let result = board.place_piece(column);
            if index + 1 < moves.len() {
                assert_eq!(result, GameResult::Ongoing, "game ended before move {index}");
            }
        }
        board
    }

    #[test]
    fn encoded_state_uses_the_side_to_move_as_the_first_plane() {
        let mut board = Board::new();
        assert_eq!(board.encode_state(), [[[0.0; 7]; 6]; 2]);

        board.place_piece(2);
        let planes = board.encode_state();
        assert_eq!(board.current_player(), Player::Yellow);
        assert_eq!(planes[0][0][2], 0.0);
        assert_eq!(planes[1][0][2], 1.0);

        board.place_piece(2);
        let planes = board.encode_state();
        assert_eq!(board.current_player(), Player::Red);
        assert_eq!(planes[0][0][2], 1.0);
        assert_eq!(planes[1][1][2], 1.0);
        assert_eq!(planes[0].iter().flatten().sum::<f32>(), 1.0);
        assert_eq!(planes[1].iter().flatten().sum::<f32>(), 1.0);
    }

    #[test]
    fn full_column_is_removed_from_valid_moves() {
        let mut board = Board::new();
        assert_eq!(board.valid_moves(), vec![0, 1, 2, 3, 4, 5, 6]);

        for _ in 0..6 {
            assert_eq!(board.place_piece(3), GameResult::Ongoing);
        }

        assert_eq!(board.valid_moves(), vec![0, 1, 2, 4, 5, 6]);
        assert_eq!(board.encode_state()[0][4][3], 1.0);
        assert_eq!(board.encode_state()[1][5][3], 1.0);
    }

    #[test]
    fn detects_wins_in_all_four_directions() {
        let wins: [&[u8]; 4] = [
            &[3, 5, 3, 6, 3, 2, 3],
            &[3, 3, 5, 3, 2, 6, 4],
            &[2, 3, 5, 2, 4, 1, 3, 1, 2, 1, 1],
            &[1, 4, 4, 4, 2, 3, 2, 3, 3, 5, 4],
        ];

        for moves in wins {
            let board = play(moves);
            assert_eq!(board.result(), GameResult::Win(Player::Red));
            assert_eq!(board.current_player(), Player::Yellow);
        }

        let yellow_win = play(&[0, 1, 0, 1, 2, 1, 2, 1]);
        assert_eq!(yellow_win.result(), GameResult::Win(Player::Yellow));
        assert_eq!(yellow_win.current_player(), Player::Red);
    }

    #[test]
    fn full_board_without_a_win_is_a_draw() {
        let moves = [
            0, 5, 4, 0, 6, 6, 2, 2, 6, 1, 6, 1, 1, 3, 5, 6, 3, 5, 0, 5, 5, 1, 1, 3,
            1, 0, 2, 3, 4, 5, 2, 3, 3, 4, 2, 0, 6, 2, 4, 0, 4, 4,
        ];
        let board = play(&moves);

        assert_eq!(board.result(), GameResult::Draw);
        assert!(board.valid_moves().is_empty());
    }
}
