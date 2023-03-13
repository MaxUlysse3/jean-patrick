use jean_patrick::{
    network::Network,
};

use rulinalg::{
    vector::Vector,
};

fn main() {
    let mut n = Network::new(256, &[16, 16], 10);
    let acts = Vector::<f64>::from_iter((0..256).map(|x| x as f64));

    let out = n.compute(acts);

    println!("{out}");
}
