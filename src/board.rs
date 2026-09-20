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

        let shift = column as u64 * 7;
        let bottom = 1u64 << shift;
        let column_mask = 0x3Fu64 << shift;
        let pos = (self.occupied() + bottom) & column_mask;
        
        debug_assert!(pos != 0, "attempted move in full column");

        match self.current_player {
            Player::Red => self.red.0 |= pos,
            Player::Yellow => self.yellow.0 |= pos,
        }

        let result = self.result();

        if matches!(result, GameResult::Ongoing) {
            self.current_player = self.current_player.other();
        }

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
        if self.has_won(self.current_player) {
            GameResult::Win(self.current_player)
        } else if self.is_draw() {
            GameResult::Draw
        } else {
            GameResult::Ongoing
        }
    }
}


impl Player {
    fn other(self) -> Player {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }
}
