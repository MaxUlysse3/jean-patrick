use rulinalg::{
    matrix::{Matrix, BaseMatrix},
    vector::Vector,
};

use libm;

use rand::Rng;

/// The struct representing a layer.
#[derive(Debug)]
pub struct Layer {
    weights: Option<Matrix<f64>>,
    biases: Option<Vector<f64>>,
    next: Option<Box<Self>>,
    size: usize,
}

/// # Constructors for [`Layer`].
impl Layer {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            weights: None,
            biases: None,
            next: None,
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

    pub fn link(&mut self, next: Box<Self>) {
        self.weights = Some(Matrix::<f64>::from_fn(next.get_size(), self.size, |_, _| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64));
        self.biases = Some(Vector::<f64>::from_fn(next.get_size(), |_| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64));

        self.next = Some(next);
    }

    pub fn set_out(&mut self, out_size: usize) {
        self.weights = Some(Matrix::<f64>::from_fn(out_size, self.size, |_, _| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64));
        self.biases = Some(Vector::<f64>::from_fn(out_size, |_| rand::thread_rng().gen_range(-1000..1000) as f64/1000f64));
    }

    #[inline]
    fn compute_acts(&self, acts: Vector<f64>) -> Vector<f64> {
        // eprintln!("{:#?}", self);
        if self.weights == None || self.biases == None {
            panic!("This `Layer` isn't linked.");
        }
        let sigma = |x: f64| 1f64/(1f64 + libm::exp(-x));
        // eprintln!("Weights: \n{} ({}, {}), \nActs: {} ({}), \nBiases: {} ({})", self.weights.as_ref().unwrap(),
        //     self.weights.as_ref().unwrap().rows(),
        //     self.weights.as_ref().unwrap().cols(),
        //     acts, acts.size(), self.biases.as_ref().unwrap(), self.biases.as_ref().unwrap().size());

        let mut new_acts = self.weights.as_ref().unwrap() * acts;
        // eprintln!("({})", new_acts.size());
        new_acts += self.biases.as_ref().unwrap();

        new_acts = new_acts.into_iter()
            .map(sigma)
            .collect();
        new_acts
    }

    #[inline]
    pub fn compute(&self, acts: Vector<f64>) -> Vector<f64> {
        if acts.size() != self.size {
            panic!("This `Layer` has a size of {} but it was fed an activation vector of size {}", self.size, acts.size());
        }
        match &self.next {
            None => self.compute_acts(acts),
            Some(next) => next.compute(self.compute_acts(acts)),
        }
    }
}
