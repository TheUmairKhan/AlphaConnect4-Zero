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

use super::minimax;

#[derive(Clone, Copy, Debug)]
pub struct MatchSummary {
    pub depth: u32,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub first_player_games: usize,
    pub first_player_wins: usize,
    pub first_player_draws: usize,
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
        }
    }

    fn record(&mut self, result: GameResult, mcts_player: Player) {
        let went_first = mcts_player == Player::Red;
        if went_first {
            self.first_player_games += 1;
        }
        match result {
            GameResult::Win(winner) if winner == mcts_player => {
                self.wins += 1;
                if went_first {
                    self.first_player_wins += 1;
                }
            }
            GameResult::Draw => {
                self.draws += 1;
                if went_first {
                    self.first_player_draws += 1;
                }
            }
            _ => self.losses += 1,
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
    let moves = AtomicUsize::new(0);
    let mut positions_evaluated = 0;
    let (request_sender, request_receiver) = mpsc::sync_channel(evaluation.threads);

    thread::scope(|scope| {
        for thread_index in 0..evaluation.threads {
            let request_sender = request_sender.clone();
            let summaries = &summaries;
            let moves = &moves;
            scope.spawn(move || {
                let queue = Rc::new(RefCell::new(Vec::new()));
                let games = (thread_index..total)
                    .step_by(evaluation.threads)
                    .map(|index| {
                        Box::pin(play_indexed_game(queue.clone(), config, index, summaries, moves))
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
    index: usize,
    summaries: &Mutex<Vec<MatchSummary>>,
    moves: &AtomicUsize,
) {
    let evaluation = &config.evaluation;
    let depth_index = index / evaluation.games;
    let mcts_player = if index % evaluation.games % 2 == 0 { Player::Red } else { Player::Yellow };
    let client = BatchingClient::new(queue);
    let result = play_game(&client, config, evaluation.depths[depth_index], mcts_player, moves).await;
    summaries.lock().unwrap()[depth_index].record(result, mcts_player);
}

async fn play_game<E: LeafEvaluator>(
    evaluator: &E,
    config: &TrainingConfig,
    depth: u32,
    mcts_player: Player,
    moves: &AtomicUsize,
) -> GameResult {
    let mut env = Connect4Env::new();
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::executor::block_on;

    use crate::{
        board::{Board, GameResult, Player},
        mcts::{evaluation::{EvalOutput, LeafEvaluator}, policy::NetworkTiming},
        config::TrainingConfig,
    };

    use super::{MatchSummary, evaluate, play_game};

    struct UniformEvaluator;

    impl LeafEvaluator for UniformEvaluator {
        async fn evaluate(&self, _board: Board) -> EvalOutput {
            EvalOutput { priors: [1.0 / 7.0; 7], value: 0.0 }
        }

        fn timing(&self) -> NetworkTiming { NetworkTiming::default() }
    }

    fn config(games: usize) -> TrainingConfig {
        let mut config: TrainingConfig = toml::from_str(include_str!("../../../training.toml")).unwrap();
        config.evaluation.games = games;
        config.evaluation.simulations = 50;
        config
    }

    fn uniform(boards: &[Board]) -> Vec<EvalOutput> {
        boards.iter().map(|_| EvalOutput { priors: [1.0 / 7.0; 7], value: 0.0 }).collect()
    }

    #[test]
    fn summary_splits_results_by_who_moved_first() {
        let mut summary = MatchSummary::new(4);
        summary.record(GameResult::Win(Player::Red), Player::Red);
        summary.record(GameResult::Draw, Player::Red);
        summary.record(GameResult::Win(Player::Red), Player::Yellow);
        summary.record(GameResult::Win(Player::Yellow), Player::Yellow);

        assert_eq!((summary.wins, summary.losses, summary.draws), (2, 1, 1));
        assert_eq!(summary.games(), 4);
        assert_eq!(summary.score_rate(), 0.625);
        assert_eq!(summary.first_player_score_rate(), 0.75);
        assert_eq!(summary.second_player_score_rate(), 0.5);
        assert!(summary.elo() > 0.0);
    }

    #[test]
    fn elo_stays_finite_for_a_clean_sweep() {
        let mut summary = MatchSummary::new(2);
        summary.record(GameResult::Win(Player::Red), Player::Red);
        summary.record(GameResult::Win(Player::Yellow), Player::Yellow);
        assert!(summary.elo().is_finite());
    }

    #[test]
    fn game_counts_every_move_played() {
        let config = config(2);
        let moves = AtomicUsize::new(0);
        block_on(play_game(&UniformEvaluator, &config, 2, Player::Red, &moves));
        assert!(moves.load(Ordering::Relaxed) >= 7);
    }

    #[test]
    fn progress_reports_grow_while_games_run() {
        let mut config = config(4);
        config.evaluation.depths = vec![1, 2];
        config.evaluation.threads = 2;

        let mut reports = Vec::new();
        let summaries = evaluate(&config, uniform, |progress| {
            reports.push((progress.games_finished, progress.moves_played, progress.total_games, progress.positions_evaluated));
        });

        assert!(!reports.is_empty());
        assert!(reports.iter().all(|&(_, _, total, _)| total == 8));
        assert!(reports.windows(2).all(|pair| pair[0].0 <= pair[1].0 && pair[0].1 <= pair[1].1 && pair[0].3 < pair[1].3));
        assert!(reports.last().unwrap().1 > 0);
        assert_eq!(summaries.iter().map(MatchSummary::games).sum::<usize>(), 8);
    }

    #[test]
    fn games_finish_with_mcts_on_either_side() {
        let config = config(2);
        for mcts_player in [Player::Red, Player::Yellow] {
            let result = block_on(play_game(&UniformEvaluator, &config, 2, mcts_player, &AtomicUsize::new(0)));
            assert_ne!(result, GameResult::Ongoing);
        }
    }

    #[test]
    fn concurrent_evaluation_plays_every_game_for_every_depth() {
        let mut config = config(5);
        config.evaluation.depths = vec![1, 2];
        config.evaluation.threads = 3;

        let summaries = evaluate(&config, uniform, |_| {});

        assert_eq!(summaries.iter().map(|s| s.depth).collect::<Vec<_>>(), [1, 2]);
        for summary in summaries {
            assert_eq!(summary.games(), 5);
            assert_eq!(summary.first_player_games, 3);
        }
    }

    #[test]
    fn finishes_when_threads_outnumber_games() {
        let mut config = config(3);
        config.evaluation.depths = vec![1];
        config.evaluation.threads = 4;

        let mut batch_sizes = Vec::new();
        let summaries = evaluate(&config, |boards| {
            batch_sizes.push(boards.len());
            uniform(boards)
        }, |_| {});

        assert_eq!(summaries[0].games(), 3);
        assert!(batch_sizes.iter().all(|&size| size > 0 && size <= 3));
    }
}
