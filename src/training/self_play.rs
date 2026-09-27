use burn::prelude::Backend;

use crate::{
    board::{GameResult, Player},
    env::Connect4Env,
    mcts::{mcts::MCTS, policy::ZeroNetPolicy},
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

pub fn self_play<B: Backend>(policy: ZeroNetPolicy<B>, simulations: u32, c_puct: f32,) -> Vec<TrainingExample> {
    let mut env = Connect4Env::new();
    let mut mcts = MCTS::new(env, policy, simulations, c_puct);
    let mut positions = Vec::new();

    let result = loop {
        let state = env.state().encode_state();
        let player = env.state().current_player();

        mcts.search();
        let policy = mcts.target_policy();
        let action = mcts.select_action() as u8;
        positions.push(PendingExample { state, policy, player });

        match env.step(action) {
            GameResult::Ongoing => {}
            result => break result,
        }
    };

    positions
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
        .collect()
}