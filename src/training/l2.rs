use burn::{
    module::{ModuleVisitor, Param},
    prelude::*,
};

struct SquaredL2<B: Backend> {
    total: Tensor<B, 1>,
}

impl<B: Backend> ModuleVisitor<B> for SquaredL2<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let weights = param.val();
        self.total = self.total.clone() + (weights.clone() * weights).sum();
    }
}

pub fn squared_l2<B: Backend>(model: &impl Module<B>, device: &B::Device) -> Tensor<B, 1> {
    let mut visitor = SquaredL2 {
        total: Tensor::zeros([1], device),
    };
    model.visit(&mut visitor);
    visitor.total
}
