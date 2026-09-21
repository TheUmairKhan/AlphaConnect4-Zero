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
        simulations: u32
    }

    impl MCTS {
        fn new(policy: StubPolicy, simulations: u32) -> Self {
            Self { 
                nodes: vec![Node::new(Connect4Env::new())],
                policy,
                simulations
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
                                self.nodes[child_idx].priors = policy;
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

                let action = self.puct(node_idx, 1.0);

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

        fn puct(&self, node_idx: usize, c: f32) -> usize {
            let node = &self.nodes[node_idx];
            let valid_moves = node.state.state().valid_moves();
            let parent_visits = self.nodes[node_idx].visits as f32;
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
                let u = c * prior * parent_visits.sqrt() / (1.0 + child_visits);
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_action = a;
                }
            }
            best_action
        }
    }
