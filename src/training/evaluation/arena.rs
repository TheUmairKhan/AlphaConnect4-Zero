use futures::executor::block_on;

use crate::{
    board::{GameResult, Player},
    config::TrainingConfig,
    env::Connect4Env,
    mcts::{evaluation::LeafEvaluator, mcts::MCTS},
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

pub fn evaluate<E: LeafEvaluator>(evaluator: &E, config: &TrainingConfig) -> Vec<MatchSummary> {
    config
        .evaluation
        .depths
        .iter()
        .map(|&depth| play_match(evaluator, config, depth))
        .collect()
}

fn play_match<E: LeafEvaluator>(evaluator: &E, config: &TrainingConfig, depth: u32) -> MatchSummary {
    let mut summary = MatchSummary::new(depth);
    for game in 0..config.evaluation.games {
        let mcts_player = if game % 2 == 0 { Player::Red } else { Player::Yellow };
        let result = block_on(play_game(evaluator, config, depth, mcts_player));
        summary.record(result, mcts_player);
    }
    summary
}

async fn play_game<E: LeafEvaluator>(
    evaluator: &E,
    config: &TrainingConfig,
    depth: u32,
    mcts_player: Player,
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

        match env.step(action) {
            GameResult::Ongoing => mcts.advance_root(action as usize).await,
            result => break result,
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use crate::{
        board::{Board, GameResult, Player},
        mcts::{evaluation::{EvalOutput, LeafEvaluator}, policy::NetworkTiming},
        config::TrainingConfig,
    };

    use super::{MatchSummary, play_game, play_match};

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
    fn games_finish_with_mcts_on_either_side() {
        let config = config(2);
        for mcts_player in [Player::Red, Player::Yellow] {
            let result = block_on(play_game(&UniformEvaluator, &config, 2, mcts_player));
            assert_ne!(result, GameResult::Ongoing);
        }
    }

    #[test]
    fn match_alternates_colors() {
        let summary = play_match(&UniformEvaluator, &config(4), 1);
        assert_eq!(summary.games(), 4);
        assert_eq!(summary.first_player_games, 2);
    }
}
