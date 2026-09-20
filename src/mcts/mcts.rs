    use crate::env::Connect4Env;
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

    struct MCTS {
        nodes: Vec<Node>,
        policy: StubPolicy
    }

    impl MCTS {
        fn new(policy: StubPolicy) -> Self {
            Self { 
                nodes: vec![Node::new(Connect4Env::new())],
                policy,
            }
        }
        fn select(&self) -> Option<(usize, usize)> {
            let mut node_idx = 0;

            loop {
                let action = self.puct(node_idx, 1.0)?;

                match self.nodes[node_idx].children[action] {
                    Some(child_idx) => {
                        // select action and traverse tree
                        node_idx = child_idx;
                    }
                    None => {
                        // Expand at leaf node
                        return Some((node_idx, action));
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

        fn backpropagate(&mut self, leaf_idx: usize) {
            let mut i: usize = leaf_idx;
            let (policy, mut _value) = self.policy.evaluate(self.nodes[i].state.state());
            self.nodes[leaf_idx].priors = policy;

            loop {
                self.nodes[i].visits += 1;
                self.nodes[i].value += _value;

                match self.nodes[i].parent {
                    Some(parent_idx) => {
                        i = parent_idx;
                        _value = -_value;
                    }
                    None => break,
                }
            }
        }

        fn puct(&self, node_idx: usize, c: f32) -> Option<usize> {
            let node = &self.nodes[node_idx];
            let valid_moves = node.state.state().valid_moves();
            let parent_visits = self.nodes[node_idx].visits as f32;
            let mut best_action = None;
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
                    best_action = Some(a);
                }
            }
            best_action
        }
    }
