use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct InputBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct InputBlockConfig {
    input_channels: usize,
    hidden_size: usize,
}

impl InputBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InputBlock<B> {
        InputBlock { 
            conv: Conv2dConfig::new([self.input_channels, self.hidden_size], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
             bn: BatchNormConfig::new(self.hidden_size).init(device),
             relu: Relu::new()
        }
    }
}

impl <B:Backend>InputBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> { // [B, 2, H, W] -> [B, hidden_size, H, W]
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}