use std::{matches, unreachable};

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Gamma};

use crate::{board::GameResult, env::Connect4Env};

use super::{evaluation::LeafEvaluator, policy::NetworkTiming};
    struct Node {
        state: Connect4Env,
        parent: Option<usize>,
        children: [Option<usize>; 7],
        visits: u32,
        value: f32, 
        priors: [f32; 7]
    }

    impl Node {
        fn new(state: Connect4Env) -> Self {
            Self {
                state,
                parent: None,
                children: [None; 7],
                visits: 0,
                value: 0.0,
                priors: [1.0 / 7.0; 7],
            }
        }
    }

    enum SelectionResult {
        Expand { leaf_idx: usize, action: usize},
        Terminal { node_idx: usize, result: GameResult}
    }

    pub struct MCTS<E: LeafEvaluator> {
        nodes: Vec<Node>,
        evaluator: E,
        simulations: u32,
        c_puct: f32,
        root_idx: usize
    }

    impl<E: LeafEvaluator> MCTS<E> {
        pub async fn new(state: Connect4Env, evaluator: E, simulations: u32, c_puct: f32) -> Self {
            let mut root = Node::new(state);
            root.priors = evaluator.evaluate(*root.state.state()).await.priors;
            Self {
                nodes: vec![root],
                evaluator,
                simulations,
                c_puct,
                root_idx: 0,
            }
        }

        pub fn network_timing(&self) -> NetworkTiming {
            self.evaluator.timing()
        }

        pub fn add_root_noise<R: Rng + ?Sized>(&mut self, gamma: &Gamma<f64>, epsilon: f32, rng: &mut R) {
            let legal_moves = self.nodes[self.root_idx].state.state().valid_moves();
            let mut noise = [0.0_f64; 7];
            let mut total = 0.0;

            for &action in &legal_moves {
                let sample = gamma.sample(rng);
                noise[action as usize] = sample;
                total += sample;
            }

            for action in legal_moves {
                let prior = &mut self.nodes[self.root_idx].priors[action as usize];
                *prior = (1.0 - epsilon) * *prior + epsilon * (noise[action as usize] / total) as f32;
            }
        }

        pub async fn select_action<R: Rng + ?Sized>(&mut self, temperature: f32, rng: &mut R) -> (usize, [f32; 7]) {
            let visit_policy = self.target_policy(1.0);
            let sampling_policy = self.target_policy(temperature);
            let draw = rng.random::<f32>();
            let mut cumulative = 0.0;
            let mut action = 0;

            for (column, &probability) in sampling_policy.iter().enumerate() {
                if probability > 0.0 {
                    action = column;
                    cumulative += probability;
                    if draw < cumulative {
                        break;
                    }
                }
            }

            self.advance_root(action).await;
            (action, visit_policy)
        }

        pub fn best_action(&self) -> usize {
            let root = &self.nodes[self.root_idx];
            (0..7)
                .filter_map(|action| root.children[action].map(|child| (action, self.nodes[child].visits)))
                .max_by_key(|&(action, visits)| (visits, std::cmp::Reverse(action)))
                .map(|(action, _)| action)
                .expect("search has not expanded any root action")
        }

        pub async fn advance_root(&mut self, action: usize) {
            let child_idx = match self.nodes[self.root_idx].children[action] {
                Some(idx) => idx,
                None => {
                    let idx = self.expand(self.root_idx, action as u8);
                    if matches!(self.nodes[idx].state.state().result(), GameResult::Ongoing) {
                        self.nodes[idx].priors = self.evaluator.evaluate(*self.nodes[idx].state.state()).await.priors;
                    }
                    idx
                }
            };
            self.nodes[child_idx].parent = None;
            self.root_idx = child_idx;
        }

        pub async fn search(&mut self) {
            for _ in 0..self.simulations {
                match self.select() {
                    SelectionResult::Expand { leaf_idx, action } => {
                        let child_idx = self.expand(leaf_idx, action as u8);
                        match self.nodes[child_idx].state.state().result() {
                            GameResult::Ongoing => {
                                let result = self.evaluator.evaluate(*self.nodes[child_idx].state.state()).await;
                                self.nodes[child_idx].priors = result.priors;
                                self.backpropagate(child_idx, result.value);
                            }
                            GameResult::Win(_) => {
                                self.backpropagate(child_idx, -1.0);
                            }
                            GameResult::Draw => {
                                self.backpropagate(child_idx, 0.0);
                            }
                        }
                    }
                    SelectionResult::Terminal { node_idx, result } => {
                        let value = match result {
                            GameResult::Win(_) => -1.0,
                            GameResult::Draw => 0.0,
                            GameResult::Ongoing => unreachable!()
                        };
                        self.backpropagate(node_idx, value);
                    }
                }
            }
        }

        fn select(&self) -> SelectionResult {
            let mut node_idx = self.root_idx;

            loop {
                let result = self.nodes[node_idx].state.state().result();
                if !matches!(result, GameResult::Ongoing) {
                    return SelectionResult::Terminal { node_idx, result };
                }

                let action = self.puct(node_idx);

                match self.nodes[node_idx].children[action] {
                    Some(child_idx) => {
                        // select action and traverse tree
                        node_idx = child_idx;
                    }
                    None => {
                        // Expand at leaf node
                        return SelectionResult::Expand { leaf_idx: node_idx, action }
                    }
                }
            }
        }

        fn expand(&mut self, leaf_idx: usize, action: u8) -> usize {
            let mut new_state = self.nodes[leaf_idx].state;
            new_state.step(action);

            let mut child = Node::new(new_state);
            child.parent = Some(leaf_idx);

            let child_idx: usize = self.nodes.len();
            self.nodes.push(child);
            self.nodes[leaf_idx].children[action as usize] = Some(child_idx);
            child_idx
        }

        fn backpropagate(&mut self, leaf_idx: usize, mut value: f32) {
            let mut i: usize = leaf_idx;

            loop {
                self.nodes[i].visits += 1;
                self.nodes[i].value += value;

                match self.nodes[i].parent {
                    Some(parent_idx) => {
                        i = parent_idx;
                        value = -value;
                    }
                    None => break,
                }
            }
        }

        fn puct(&self, node_idx: usize) -> usize {
            let node = &self.nodes[node_idx];
            let valid_moves = node.state.state().valid_moves();
            let parent_visits = self.nodes[node_idx].visits.max(1) as f32;
            let mut best_action = valid_moves[0] as usize;
            let mut best_score = f32::NEG_INFINITY;

            for action in valid_moves {
                let a = action as usize;
                let prior = node.priors[a];

                let (q, child_visits) = match node.children[a] {
                    Some(child_idx) => {
                        let child = &self.nodes[child_idx];
                        let q = if child.visits > 0 {
                            -(child.value / child.visits as f32)
                        } else {
                            0.0
                        };
                        (q, child.visits as f32)
                    }
                    None => (0.0, 0.0),
                };
                let u = self.c_puct * prior * parent_visits.sqrt() / (1.0 + child_visits);
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_action = a;
                }
            }
            best_action
        }

        // probability distribution of real action visits for a given state
        pub fn target_policy(&self, temperature: f32) -> [f32; 7] {
            let root = &self.nodes[self.root_idx];
            let mut pi = [0.0; 7];
            let max_visits = root.children
                .iter()
                .filter_map(|child| child.map(|idx| self.nodes[idx].visits))
                .max()
                .unwrap_or(0);

            if max_visits == 0 {
                return pi;
            }

            let mut total = 0.0;
            for action in 0..7 {
                if let Some(child_idx) = root.children[action] {
                    let visits = self.nodes[child_idx].visits as f32;
                    pi[action] = (visits / max_visits as f32).powf(1.0 / temperature);
                    total += pi[action];
                }
            }
            for probability in &mut pi {
                *probability /= total;
            }
            pi
        }
    }

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use futures::executor::block_on;

    use crate::{board::Board, env::Connect4Env};

    use super::{super::{evaluation::{EvalOutput, LeafEvaluator}, policy::NetworkTiming}, MCTS};

    struct CountingEvaluator(Rc<Cell<usize>>);

    impl LeafEvaluator for CountingEvaluator {
        async fn evaluate(&self, _board: Board) -> EvalOutput {
            self.0.set(self.0.get() + 1);
            EvalOutput { priors: [1.0 / 7.0; 7], value: 0.25 }
        }

        fn timing(&self) -> NetworkTiming { NetworkTiming::default() }
    }

    #[test]
    fn root_leaf_and_unexpanded_next_root_use_the_evaluator() {
        let calls = Rc::new(Cell::new(0));
        let mut search = block_on(MCTS::new(Connect4Env::new(), CountingEvaluator(calls.clone()), 1, 1.0));
        assert_eq!(calls.get(), 1);

        block_on(search.search());
        assert_eq!(calls.get(), 2);

        block_on(search.advance_root(1));
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn select_action_returns_raw_visit_distribution_regardless_of_temperature() {
        let calls = Rc::new(Cell::new(0));
        let mut search = block_on(MCTS::new(Connect4Env::new(), CountingEvaluator(calls), 50, 1.0));
        block_on(search.search());

        let root = &search.nodes[search.root_idx];
        let visits: Vec<f32> = root.children
            .iter()
            .map(|child| child.map_or(0.0, |idx| search.nodes[idx].visits as f32))
            .collect();
        let total: f32 = visits.iter().sum();

        let (_, policy) = block_on(search.select_action(0.1, &mut rand::rng()));

        for (probability, visits) in policy.iter().zip(visits) {
            assert!((probability - visits / total).abs() < 1e-6);
        }
    }
}
