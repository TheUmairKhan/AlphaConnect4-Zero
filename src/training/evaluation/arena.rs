use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use crate::{
    board::{Board, GameResult, Player},
    config::TrainingConfig,
    env::Connect4Env,
    mcts::{
        evaluation::{BatchingClient, EvalOutput, LeafEvaluator},
        mcts::MCTS,
    },
    training::accelerator::{
        accelerator,
        worker::{LeafQueue, drive_games},
    },
};

use super::{minimax, openings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    Draw,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Win => "win",
            Outcome::Loss => "loss",
            Outcome::Draw => "draw",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MatchSummary {
    pub depth: u32,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub first_player_games: usize,
    pub first_player_wins: usize,
    pub first_player_draws: usize,
    pub empty_board_first: Option<Outcome>,
    pub empty_board_second: Option<Outcome>,
}

impl MatchSummary {
    fn new(depth: u32) -> Self {
        Self {
            depth,
            wins: 0,
            losses: 0,
            draws: 0,
            first_player_games: 0,
            first_player_wins: 0,
            first_player_draws: 0,
            empty_board_first: None,
            empty_board_second: None,
        }
    }

    fn record(&mut self, result: GameResult, mcts_player: Player, from_empty_board: bool) {
        let went_first = mcts_player == Player::Red;
        let outcome = match result {
            GameResult::Win(winner) if winner == mcts_player => Outcome::Win,
            GameResult::Draw => Outcome::Draw,
            _ => Outcome::Loss,
        };
        if went_first {
            self.first_player_games += 1;
        }
        match outcome {
            Outcome::Win => {
                self.wins += 1;
                if went_first {
                    self.first_player_wins += 1;
                }
            }
            Outcome::Draw => {
                self.draws += 1;
                if went_first {
                    self.first_player_draws += 1;
                }
            }
            Outcome::Loss => self.losses += 1,
        }
        if from_empty_board {
            if went_first {
                self.empty_board_first = Some(outcome);
            } else {
                self.empty_board_second = Some(outcome);
            }
        }
    }

    pub fn games(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    pub fn win_rate(&self) -> f32 {
        rate(self.wins as f32, self.games())
    }

    pub fn loss_rate(&self) -> f32 {
        rate(self.losses as f32, self.games())
    }

    pub fn draw_rate(&self) -> f32 {
        rate(self.draws as f32, self.games())
    }

    pub fn score_rate(&self) -> f32 {
        rate(self.wins as f32 + 0.5 * self.draws as f32, self.games())
    }

    pub fn first_player_score_rate(&self) -> f32 {
        rate(
            self.first_player_wins as f32 + 0.5 * self.first_player_draws as f32,
            self.first_player_games,
        )
    }

    pub fn second_player_score_rate(&self) -> f32 {
        let games = self.games() - self.first_player_games;
        let wins = self.wins - self.first_player_wins;
        let draws = self.draws - self.first_player_draws;
        rate(wins as f32 + 0.5 * draws as f32, games)
    }

    pub fn elo(&self) -> f32 {
        let games = self.games();
        if games == 0 {
            return 0.0;
        }
        let margin = 0.5 / games as f32;
        let score = self.score_rate().clamp(margin, 1.0 - margin);
        -400.0 * (1.0 / score - 1.0).log10()
    }
}

fn rate(points: f32, games: usize) -> f32 {
    if games == 0 { 0.0 } else { points / games as f32 }
}

pub struct EvaluationProgress {
    pub games_finished: usize,
    pub total_games: usize,
    pub moves_played: usize,
    pub positions_evaluated: usize,
    pub summaries: Vec<MatchSummary>,
}

pub fn evaluate<F, P>(config: &TrainingConfig, mut evaluate_batch: F, mut on_progress: P) -> Vec<MatchSummary>
where
    F: FnMut(&[Board]) -> Vec<EvalOutput>,
    P: FnMut(&EvaluationProgress),
{
    let evaluation = &config.evaluation;
    let summaries = Mutex::new(evaluation.depths.iter().map(|&depth| MatchSummary::new(depth)).collect::<Vec<_>>());
    let total = evaluation.depths.len() * evaluation.games;
    let openings = openings::suite(evaluation.games / 2);
    let moves = AtomicUsize::new(0);
    let mut positions_evaluated = 0;
    let (request_sender, request_receiver) = mpsc::sync_channel(evaluation.threads);

    thread::scope(|scope| {
        for thread_index in 0..evaluation.threads {
            let request_sender = request_sender.clone();
            let summaries = &summaries;
            let moves = &moves;
            let openings = &openings;
            scope.spawn(move || {
                let queue = Rc::new(RefCell::new(Vec::new()));
                let games = (thread_index..total)
                    .step_by(evaluation.threads)
                    .map(|index| {
                        Box::pin(play_indexed_game(queue.clone(), config, openings, index, summaries, moves))
                            as Pin<Box<dyn Future<Output = ()>>>
                    })
                    .collect();
                drive_games(games, &queue, &request_sender);
            });
        }
        drop(request_sender);
        accelerator::serve_requests(request_receiver, evaluation.threads, |boards| {
            let outputs = evaluate_batch(boards);
            positions_evaluated += boards.len();
            let summaries = summaries.lock().unwrap().clone();
            on_progress(&EvaluationProgress {
                games_finished: summaries.iter().map(MatchSummary::games).sum(),
                total_games: total,
                moves_played: moves.load(Ordering::Relaxed),
                positions_evaluated,
                summaries,
            });
            outputs
        });
    });

    summaries.into_inner().unwrap()
}

async fn play_indexed_game(
    queue: Rc<LeafQueue>,
    config: &TrainingConfig,
    openings: &[Vec<u8>],
    index: usize,
    summaries: &Mutex<Vec<MatchSummary>>,
    moves: &AtomicUsize,
) {
    let evaluation = &config.evaluation;
    let depth_index = index / evaluation.games;
    let game_index = index % evaluation.games;
    let opening = &openings[game_index / 2];
    let mcts_player = if game_index % 2 == 0 { Player::Red } else { Player::Yellow };
    let client = BatchingClient::new(queue);
    let result = play_game(&client, config, opening, evaluation.depths[depth_index], mcts_player, moves).await;
    summaries.lock().unwrap()[depth_index].record(result, mcts_player, opening.is_empty());
}

async fn play_game<E: LeafEvaluator>(
    evaluator: &E,
    config: &TrainingConfig,
    opening: &[u8],
    depth: u32,
    mcts_player: Player,
    moves: &AtomicUsize,
) -> GameResult {
    let mut env = Connect4Env::new();
    for &column in opening {
        env.step(column);
    }
    let mut mcts = MCTS::new(env, evaluator, config.evaluation.simulations, config.search.c_puct).await;

    loop {
        let action = if env.state().current_player() == mcts_player {
            mcts.search().await;
            mcts.best_action() as u8
        } else {
            minimax::best_action(env.state(), depth)
        };

        moves.fetch_add(1, Ordering::Relaxed);
        match env.step(action) {
            GameResult::Ongoing => mcts.advance_root(action as usize).await,
            result => break result,
        }
    }
}
