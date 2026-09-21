    use std::{matches, unreachable};

use crate::{board::GameResult, env::Connect4Env};
    use super::policy::StubPolicy;

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

    struct MCTS {
        nodes: Vec<Node>,
        policy: StubPolicy,
        simulations: u32,
        c_puct: f32,
    }

    impl MCTS {
        fn new(state: Connect4Env, policy: StubPolicy, simulations: u32, c_puct: f32) -> Self {
        let mut root = Node::new(state);
        let (priors, _value) = policy.evaluate(root.state.state());
        root.priors = Self::normalize_priors(priors, &root.state.state().valid_moves());

        Self {
            nodes: vec![root],
            policy,
            simulations,
            c_puct
        }
    }

        fn search(&mut self) {
            for _ in 0..self.simulations {
                match self.select() {
                    SelectionResult::Expand { leaf_idx, action } => {
                        let child_idx = self.expand(leaf_idx, action as u8);
                        match self.nodes[child_idx].state.state().result() {
                            GameResult::Ongoing => {
                                let (policy, value) = self.policy.evaluate(self.nodes[child_idx].state.state());
                                let valid_moves = self.nodes[child_idx].state.state().valid_moves();
                                self.nodes[child_idx].priors = Self::normalize_priors(policy, &valid_moves);
                                self.backpropagate(child_idx, value);
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
            let mut node_idx = 0;

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

        fn normalize_priors(priors: [f32; 7], valid_moves: &[u8]) -> [f32; 7] {
            let mut masked_priors = [0.0; 7];
            let mut total = 0.0;

            for &action in valid_moves {
                let prior = priors[action as usize];
                masked_priors[action as usize] = prior;
                total += prior;
            }

            if total > 0.0 {
                for prior in &mut masked_priors {
                    *prior /= total;
                }
            } else {
                let uniform_prior = 1.0 / valid_moves.len() as f32;
                for &action in valid_moves {
                    masked_priors[action as usize] = uniform_prior;
                }
            }

            masked_priors
        }

        fn best_action(&self) -> usize {
            let root = &self.nodes[0];
            root.children
                .iter()
                .enumerate()
                .filter_map(|(action, child)| {
                    child.map(|idx| (action, self.nodes[idx].visits))
                })
                .max_by_key(|&(_, visits)| visits)
                .map(|(action, _)| action)
                .expect("no valid actions")
        }

        fn root_policy(&self) -> [f32; 7] {
            let root = &self.nodes[0];
            let mut pi = [0.0; 7];
            let total_visits: u32 = root.children
                .iter()
                .filter_map(|child| {
                    child.map(|idx| self.nodes[idx].visits)
                })
                .sum();

            if total_visits == 0 {
                return pi;
            }

            for action in 0..7 {
                if let Some(child_idx) = root.children[action] {
                    pi[action] = self.nodes[child_idx].visits as f32 / total_visits as f32;
                }
            }
            pi
        }
    }
