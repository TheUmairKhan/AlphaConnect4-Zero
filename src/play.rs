use std::{
    error::Error,
    io::{self, Write},
    process,
};

use burn::{prelude::*, record::DefaultRecorder};
use futures::executor::block_on;

use crate::{
    board::{Board, GameResult, Player},
    config::TrainingConfig,
    env::Connect4Env,
    mcts::{evaluation::LeafEvaluator, mcts::MCTS, policy::ZeroNetPolicy},
    model::zeronet::ZeroNetConfig,
};

pub fn run<B: Backend>(config: &TrainingConfig, device: &B::Device) -> Result<(), Box<dyn Error>> {
    let path = &config.play.model_path;
    let model = ZeroNetConfig::new(2, config.model.hidden_size, config.model.num_res_blocks)
        .init::<B>(device)
        .load_file(path, &DefaultRecorder::new(), device)
        .map_err(|error| format!("failed to load model {path}: {error}"))?;
    println!("Loaded {path} ({} simulations per move)", config.play.simulations);

    let human = loop {
        match prompt("Play as (r)ed, who moves first, or (y)ellow? ").as_str() {
            "r" | "red" => break Player::Red,
            "y" | "yellow" => break Player::Yellow,
            _ => println!("Enter r or y."),
        }
    };
    block_on(play_game(ZeroNetPolicy::new(model, device.clone()), config, human));
    Ok(())
}

async fn play_game<E: LeafEvaluator>(evaluator: E, config: &TrainingConfig, human: Player) {
    let mut env = Connect4Env::new();
    let mut mcts = MCTS::new(env, evaluator, config.play.simulations, config.search.c_puct).await;

    loop {
        print!("\n{}", render(env.state()));
        let action = if env.state().current_player() == human {
            loop {
                match parse_column(env.state(), &prompt("Your move (1-7, q to quit): ")) {
                    Ok(column) => break column,
                    Err(message) => println!("{message}"),
                }
            }
        } else {
            println!("Model is thinking...");
            mcts.search().await;
            let column = mcts.best_action() as u8;
            println!("Model plays column {}.", column + 1);
            column
        };

        match env.step(action) {
            GameResult::Ongoing => mcts.advance_root(action as usize).await,
            result => {
                print!("\n{}", render(env.state()));
                match result {
                    GameResult::Win(winner) if winner == human => println!("You win!"),
                    GameResult::Win(_) => println!("Model wins."),
                    _ => println!("Draw."),
                }
                return;
            }
        }
    }
}

fn prompt(message: &str) -> String {
    print!("{message}");
    io::stdout().flush().unwrap();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).unwrap() == 0 {
        process::exit(0);
    }
    let line = line.trim().to_ascii_lowercase();
    if line == "q" || line == "quit" {
        process::exit(0);
    }
    line
}

fn parse_column(board: &Board, line: &str) -> Result<u8, &'static str> {
    let column = match line.parse::<u8>() {
        Ok(number @ 1..=7) => number - 1,
        _ => return Err("Enter a column from 1 to 7."),
    };
    if board.valid_moves().contains(&column) {
        Ok(column)
    } else {
        Err("That column is full.")
    }
}

fn render(board: &Board) -> String {
    let red = board.pieces(Player::Red);
    let yellow = board.pieces(Player::Yellow);
    let mut text = String::new();
    for row in (0..6).rev() {
        text.push('|');
        for column in 0..7 {
            let bit = 1u64 << (column * 7 + row);
            text.push_str(if red & bit != 0 {
                " \x1b[31m●\x1b[0m"
            } else if yellow & bit != 0 {
                " \x1b[33m●\x1b[0m"
            } else {
                " ·"
            });
        }
        text.push_str(" |\n");
    }
    text.push_str("  1 2 3 4 5 6 7\n");
    text
}

#[cfg(test)]
mod tests {
    use crate::board::Board;

    use super::{parse_column, render};

    #[test]
    fn columns_are_one_based_and_must_have_room() {
        let mut board = Board::new();
        assert_eq!(parse_column(&board, "1"), Ok(0));
        assert_eq!(parse_column(&board, "7"), Ok(6));
        assert!(parse_column(&board, "0").is_err());
        assert!(parse_column(&board, "8").is_err());
        assert!(parse_column(&board, "abc").is_err());
        for _ in 0..6 {
            board.place_piece(3);
        }
        assert_eq!(parse_column(&board, "4"), Err("That column is full."));
    }

    #[test]
    fn board_renders_pieces_bottom_up() {
        let mut board = Board::new();
        board.place_piece(0);
        board.place_piece(0);
        let lines: Vec<String> = render(&board).lines().map(String::from).collect();
        assert_eq!(lines.len(), 7);
        assert!(lines[5].starts_with("| \x1b[31m●"));
        assert!(lines[4].starts_with("| \x1b[33m●"));
        assert_eq!(lines[6], "  1 2 3 4 5 6 7");
    }
}
