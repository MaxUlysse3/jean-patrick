/// The module containing layers.
mod layer;

use rulinalg::{
    vector::Vector,
};

use layer::Layer;

/// The struct representing a neural network.
#[derive(Debug)]
pub struct Network {
    input: Layer,
    output: *const Layer,
}

/// # Constructors for [`Network`].
impl Network {
    /// Return a new [`Network`] with an `in_size` sized input layer, an `out_size` sized output layer and `hidden_sizes.len()` hidden_layers with respective sizes.
    #[inline]
    pub fn new(in_size: usize, hidden_sizes: &[usize], out_size: usize) -> Self {
        let mut input = Layer::new(in_size);
        let mut last = &mut input as *mut Layer;

        for i in hidden_sizes {
            let mut layer = Box::new(Layer::new(*i));
            let layer_ptr = layer.as_mut() as *mut Layer;
            unsafe {
                (&mut *last).link(layer);
            }
            // eprintln!("{:?}", unsafe {&*last});
            last = layer_ptr;
        }

        unsafe {
            (&mut *last).set_out(out_size);
        }

        Self {
            input,
            output: last as *const Layer,
        }
    }
}

/// # Associated functions for [`Network`].
impl Network {
    /// Compute the output of the network with given activation.
    #[inline]
    pub fn compute(&self, acts: Vector<f64>) -> Vector<f64> {
        self.input.compute(acts)
    }
}
