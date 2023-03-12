use rulinalg::{
    matrix::Matrix,
    vector::Vector,
};

use rand::Rng;

/// The struct representing a layer.
pub struct Layer {
    weights: Option<Matrix<f64>>,
    act: Option(Vector<f64>),
    biases: Vector<f64>,
    prec: Option<*const Self>,
    size: usize,
}

/// # Constructors for [`Layer`].
impl Layer {
    #[inline]
    pub fn new(size: usize, is_input: bool) -> Self {
        let weights = None;
        let act = match is_input {
            true => Some(Vector::<f64>::from_fn(size, |_| 0f64)),
            false => None,
        };
        let biases = Vector::<f64>::from_fn(size, |_| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64);

        Self {
            weights: weights,
            act: act,
            biases: biases,
            prec: Some(0 as *const Layer),
            size,
        }
    }
}

/// # Associated functions for [`Layer`].
impl Layer {
    #[inline]
    pub fn get_size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn get_act(&self) -> &Vector<f64> {
        &self.act
    }

    #[inline]
    pub fn get_act_mut(&mut self) -> &mut Vector<f64> {
        &mut self.act
    }

    pub fn link(&mut self, prec: &Self) {
        self.weights = Some(Matrix::<f64>::from_fn(prec.get_size(), self.size, |_, _| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64));
        self.prec = Some(prec as *const Layer);
    }

    // TODO comput func -> maybe differents type for input layer and normal
    // pub fn compute(&self) -> Vector<f64> {
    // }
}
