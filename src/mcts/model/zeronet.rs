use crate::mcts::model::policy_head::{PolicyHead, PolicyHeadConfig};
use crate::mcts::model::resblock::{ResBlock, ResBlockConfig};
use crate::mcts::model::value_head::{ValueHead, ValueHeadConfig};
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ZeroNet<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    relu: Relu,
    res_blocks: Vec<ResBlock<B>>,
    policy_head: PolicyHead<B>,
    value_head: ValueHead<B>,
}

#[derive(Config, Debug)]
pub struct ZeroNetConfig {
    input_channels: usize,
    hidden_size: usize,
    num_res_blocks: usize,
}

impl ZeroNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZeroNet<B> {
        ZeroNet {
            conv: Conv2dConfig::new([self.input_channels, self.hidden_size], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn: BatchNormConfig::new(self.hidden_size).init(device),
            relu: Relu::new(),
            res_blocks: (0..self.num_res_blocks)
                .map(|_| {ResBlockConfig::new(self.hidden_size).init(device)})
                .collect(),
            policy_head: PolicyHeadConfig::new(self.hidden_size).init(device),
            value_head: ValueHeadConfig::new(self.hidden_size).init(device),
        }
    }
}

impl<B: Backend> ZeroNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        let x = self.res_blocks
            .iter()
            .fold(x, |x, block| block.forward(x));
        let policy = self.policy_head.forward(x.clone());
        let value = self.value_head.forward(x);
        (policy, value)
    }
}
