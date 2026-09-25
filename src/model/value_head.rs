use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    relu: Relu,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ValueHeadConfig {
    hidden_size: usize,
}

impl ValueHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            conv: Conv2dConfig::new([self.hidden_size, 1], [1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn: BatchNormConfig::new(1).init(device),
            relu: Relu::new(),
            linear: LinearConfig::new(42, 1).init(device), // 6 * 7 -> scalar
        }
    }
}

impl<B: Backend> ValueHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        let x = x.flatten(1, 3);
        self.linear.forward(x).tanh()
    }
}
