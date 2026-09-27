use burn::prelude::Backend;

use crate::{
    board::{GameResult, Player},
    env::Connect4Env,
    mcts::{mcts::MCTS, policy::ZeroNetPolicy},
    model::zeronet::ZeroNet,
    training::config::TrainingConfig,
};

pub struct EvaluationResult {
    pub candidate_wins: usize,
    pub best_wins: usize,
    pub draws: usize,
}

pub fn evaluate<B: Backend>(
    candidate: &ZeroNet<B>,
    best: &ZeroNet<B>,
    config: &TrainingConfig,
    device: &B::Device,
) -> EvaluationResult {
    let mut result = EvaluationResult {
        candidate_wins: 0,
        best_wins: 0,
        draws: 0,
    };

    for game in 0..config.evaluation.games {
        let candidate_is_red = game % 2 == 0;
        let mut env = Connect4Env::new();
        let mut candidate_search = MCTS::new(
            env,
            ZeroNetPolicy::new(candidate.clone(), device.clone()),
            config.evaluation.simulations,
            config.search.c_puct,
        );
        let mut best_search = MCTS::new(
            env,
            ZeroNetPolicy::new(best.clone(), device.clone()),
            config.evaluation.simulations,
            config.search.c_puct,
        );
        let mut rng = rand::rng();

        let outcome = loop {
            let candidate_to_move = (env.state().current_player() == Player::Red) == candidate_is_red;
            let (current, other) = if candidate_to_move {
                (&mut candidate_search, &mut best_search)
            } else {
                (&mut best_search, &mut candidate_search)
            };
            current.search();
            let (action, _) = current.select_action(0.0, &mut rng);
            other.advance_root(action);

            match env.step(action as u8) {
                GameResult::Ongoing => {}
                outcome => break outcome,
            }
        };

        match outcome {
            GameResult::Win(winner) if (winner == Player::Red) == candidate_is_red => {
                result.candidate_wins += 1;
            }
            GameResult::Win(_) => result.best_wins += 1,
            GameResult::Draw => result.draws += 1,
            GameResult::Ongoing => unreachable!(),
        }
    }

    result
}
