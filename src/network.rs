/// The module containing layers.
mod layer;

use layer::Layer;

/// The struct representing a neural network.
pub struct Network {
    input: Layer,
    output: *const Layer,
}

/// # Constructors for [`Network`].
impl Network {
    /// Return a new [`Network`] with an `in_size` sized input layer, an `out_size` sized output layer and `hidden_sizes.len()` hidden_layers with respective sizes.
    #[inline]
    pub fn new(in_size: usize, hidden_sizes: &[usize], out_size: usize) -> Self {
        let input = Layer::new(in_size, true);
        let mut to_return = Self {
            output: &input as *const Layer,
            input: input,
        };

        for i in hidden_sizes {
            let mut layer = Layer::new(*i, false);
            unsafe {
                layer.link(&*to_return.output);
            }
            to_return.output = &layer as *const Layer;
        }

        let mut out = Layer::new(out_size, false);
        unsafe {
            out.link(&*to_return.output);
        }
        to_return.output = &out as *const Layer;

        to_return
    }
}
