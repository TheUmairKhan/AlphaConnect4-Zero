use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};


#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    hidden_size: usize,
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            conv1: Conv2dConfig::new([self.hidden_size, self.hidden_size], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.hidden_size).init(device),
            conv2: Conv2dConfig::new([self.hidden_size, self.hidden_size], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.hidden_size).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone(); // [B, hidden_size, H, W]
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = x + residual;
        self.relu.forward(x)
    }
}