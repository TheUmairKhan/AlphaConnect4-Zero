use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    relu: Relu,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct PolicyHeadConfig {
    hidden_size: usize
}

impl PolicyHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            conv: Conv2dConfig::new([self.hidden_size, 1], [1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn: BatchNormConfig::new(1).init(device),
            relu: Relu::new(),
            linear: LinearConfig::new(42, 7).init(device), // 6 * 7
        }
    }
}

impl <B:Backend> PolicyHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        let x = x.flatten(1, 3);
        self.linear.forward(x)
    }
}
