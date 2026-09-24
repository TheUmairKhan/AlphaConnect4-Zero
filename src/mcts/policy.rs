use crate::board::Board;

pub struct StubPolicy;

impl StubPolicy {
    pub fn evaluate(&self, _state: &Board) -> ([f32; 7], f32) {
        ([1.0 / 7.0; 7], 0.0)
    }
}
