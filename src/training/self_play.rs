use burn::prelude::Backend;
use rand_distr::{Gamma, GammaError};

use crate::{
    board::{GameResult, Player},
    env::Connect4Env,
    mcts::{mcts::MCTS, policy::ZeroNetPolicy},
    training::config::SelfPlayConfig,
};

pub struct TrainingExample {
    pub state: [[[f32; 7]; 6]; 2],
    pub policy: [f32; 7],
    pub value: f32,
}

struct PendingExample {
    state: [[[f32; 7]; 6]; 2],
    policy: [f32; 7],
    player: Player,
}

pub fn self_play<B: Backend>(
    policy: ZeroNetPolicy<B>,
    config: &SelfPlayConfig,
) -> Result<Vec<TrainingExample>, GammaError> {
    let gamma = Gamma::new(config.dirichlet_alpha as f64, 1.0)?;
    let mut rng = rand::rng();
    let mut env = Connect4Env::new();
    let mut mcts = MCTS::new(env, policy, config.simulations, config.c_puct);
    let mut positions = Vec::new();

    let result = loop {
        let state = env.state().encode_state();
        let player = env.state().current_player();

        mcts.add_root_noise(&gamma, config.dirichlet_epsilon, &mut rng);
        mcts.search();
        let (action, policy) = mcts.select_action(config.temperature(positions.len()), &mut rng);
        positions.push(PendingExample { state, policy, player });

        match env.step(action as u8) {
            GameResult::Ongoing => {}
            result => break result,
        }
    };

    Ok(positions
        .into_iter()
        .map(|position| {
            let value = match result {
                GameResult::Win(winner) if winner == position.player => 1.0,
                GameResult::Win(_) => -1.0,
                GameResult::Draw => 0.0,
                GameResult::Ongoing => unreachable!(),
            };
            TrainingExample {
                state: position.state,
                policy: position.policy,
                value,
            }
        })
        .collect())
}