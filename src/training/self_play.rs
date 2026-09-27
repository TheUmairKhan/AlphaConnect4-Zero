use std::time::{Duration, Instant};

use burn::prelude::Backend;
use rand_distr::{Gamma, GammaError};

use crate::{
    board::{GameResult, Player},
    env::Connect4Env,
    mcts::{mcts::MCTS, policy::{NetworkTiming, ZeroNetPolicy}},
    training::config::TrainingConfig,
};

pub struct TrainingExample {
    pub state: [[[f32; 7]; 6]; 2],
    pub policy: [f32; 7],
    pub value: f32,
}

pub struct MoveTiming {
    pub total: Duration,
    pub search: Duration,
    pub search_network: Duration,
    pub network_calls: usize,
}

pub struct GameTiming {
    pub total: Duration,
    pub moves: Vec<MoveTiming>,
    pub network: NetworkTiming,
}

pub struct SelfPlayGame {
    pub examples: Vec<TrainingExample>,
    pub timing: GameTiming,
}

struct PendingExample {
    state: [[[f32; 7]; 6]; 2],
    policy: [f32; 7],
    player: Player,
}

pub fn self_play<B: Backend>(
    policy: ZeroNetPolicy<B>,
    config: &TrainingConfig,
) -> Result<SelfPlayGame, GammaError> {
    let game_started = Instant::now();
    let self_play = &config.self_play;
    let gamma = Gamma::new(self_play.dirichlet_alpha as f64, 1.0)?;
    let mut rng = rand::rng();
    let mut env = Connect4Env::new();
    let mut mcts = MCTS::new(env, policy, self_play.simulations, config.search.c_puct);
    let mut positions = Vec::new();
    let mut move_timings = Vec::new();

    let result = loop {
        let move_started = Instant::now();
        let calls_before = mcts.network_timing().calls;
        let state = env.state().encode_state();
        let player = env.state().current_player();

        mcts.add_root_noise(&gamma, self_play.dirichlet_epsilon, &mut rng);
        let network_before_search = mcts.network_timing().total;
        let search_started = Instant::now();
        mcts.search();
        let search = search_started.elapsed();
        let search_network = mcts.network_timing().total - network_before_search;
        let (action, policy) = mcts.select_action(self_play.temperature(positions.len()), &mut rng);
        positions.push(PendingExample { state, policy, player });

        let outcome = env.step(action as u8);
        move_timings.push(MoveTiming {
            total: move_started.elapsed(),
            search,
            search_network,
            network_calls: mcts.network_timing().calls - calls_before,
        });
        match outcome {
            GameResult::Ongoing => {}
            result => break result,
        }
    };

    let network = mcts.network_timing();
    let examples = positions
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
        .collect();
    Ok(SelfPlayGame {
        examples,
        timing: GameTiming {
            total: game_started.elapsed(),
            moves: move_timings,
            network,
        },
    })
}
