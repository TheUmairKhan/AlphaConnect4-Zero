struct BitBoard(pub u64);
struct Board {
    red: BitBoard,
    yellow: BitBoard,
    current_player: Player,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Player {
    Red,
    Yellow,
}

enum GameResult {
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

    fn new() -> Self {
        Self{
            red: BitBoard(0),
            yellow: BitBoard(0),
            current_player: Player::Red,
        }
    }

    fn occupied(&self) -> u64 {
        self.red.0 | self.yellow.0
    }

    fn valid_moves(&self) -> u64 {
        (self.occupied() + Self::BOTTOM_MASK) & Self::BOARD_MASK
    }

    fn place_piece(&mut self, column: u64) -> bool {
        let occupied = self.occupied();
        let bottom = 1u64 << column;
        let column_mask = 0x3Fu64 << column;
        let pos = (occupied + bottom) & column_mask;
        if pos == 0 {
            return false;   
        }
        match self.current_player {
            Player::Red => self.red.0 |= pos,
            Player::Yellow => self.yellow.0 |= pos,
        }

        self.current_player = self.current_player.other();
        true
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
        self.valid_moves() == 0
    }

    fn result(&self) -> GameResult {
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
}


impl Player {
    fn other(self) -> Player {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }
}

fn main() {
    println!("Hello, world!");
}
