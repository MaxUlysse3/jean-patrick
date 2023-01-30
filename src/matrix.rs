use std::fmt::{Display, Debug, Formatter};
use std::fmt;
use std::error::Error as StdError;
use std::alloc::{alloc, dealloc, Layout};
use std::ops::Drop;
use std::mem::{size_of, align_of};

use ndarray::{Array2};
use num_traits::identities::{Zero, One};

/// A 2 dimensional matrix containing items of type T.
/// 
/// # Exemples
///
/// ```
/// use jean_patrick::matrix::Matrix;
///
/// let matrix = Matrix::<i32>::new(3, 3, 2);
///
/// assert_eq!(matrix.get_size_i(), 3);
/// assert_eq!(matrix.get_size_j(), 3);
/// ```
pub struct Matrix<T> {
    value: *mut T,
    size_i: usize,
    size_j: usize,
}

/// # Constructors for `Matrix<T>`
impl<T> Matrix<T> {
    /// Create a new `Matrix` filled with `value`
    pub fn new(size_i: usize, size_j: usize, value: T) -> Self where
    T: Clone {

        let mut to_return = Self {
            value: Self::allocate(size_i, size_j),
            size_i: size_i,
            size_j: size_j,
        };
        
        for i in 0..to_return.size_i * to_return.size_j {
            unsafe {
                *to_return.get_ptr_to(i) = value.clone();
            }
        }
        to_return
    }

    // /// Create a new `Matrix` from default value
    // pub fn new_default(size_i: usize, size_j: usize) -> Self where
    // T: Default {
    //     Self {
    //         value: Array2::<T>::default((size_i, size_j)),
    //     }
    // }

    // /// Create a new identity `Matrix`
    // pub fn new_id(size: usize) -> Self where
    // T: Clone + Zero + One {
    //     Self {
    //         value: Array2::<T>::eye(size),
    //     }
    // }
    
    fn allocate(size_i: usize, size_j: usize) -> *mut T {
        let layout = unsafe { Layout::from_size_align_unchecked(size_of::<T>() * size_i * size_j, align_of::<T>()) };
        unsafe {
            alloc(layout) as *mut T
        }
    }

    fn get_ptr_to(&mut self, index: usize) -> *mut T {
        (self.value as usize + (index * size_of::<T>())) as *mut T
    }
}

// /// # Getters and Setters for `Matrix<T>`
// impl<T> Matrix<T> {
//     /// Return a reference to the value at index i, j of `self`
//     pub fn get(&self, i: usize, j: usize) -> Option<&T> {
//         self.value.get((i, j))
//     }

//     /// Return a mutable reference to the value at index i, j of `self`
//     pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
//         self.value.get_mut((i, j))
//     }

//     /// Return the I size of `self`
//     pub fn get_size_i(&self) -> usize {
//         self.value.shape()[0]
//     }

//     /// Return the J size of `self`
//     pub fn get_size_j(&self) -> usize {
//         self.value.shape()[1]
//     }
// }

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.value as *mut u8, unsafe { Layout::from_size_align_unchecked(size_of::<T>(), align_of::<T>()) });
        }
        drop(self);
    }
}

/// An struct representing an error
#[derive(Debug)]
pub struct Error {
    pub value: String,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl StdError for Error {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn constructors_work() {
        let m1 = Matrix::<i32>::new(3, 3, 2);
        let m2 = Matrix::<i32>::new_default(3, 3);
        let m3 = Matrix::<i32>::new_id(3);

        assert_eq!(*m1.get(1, 1).unwrap(), 2);
        assert_eq!(*m2.get(1, 1).unwrap(), i32::default());
        assert_eq!(*m3.get(1, 1).unwrap(), 1);
        assert_eq!(*m3.get(1, 2).unwrap(), 0);
    }

    #[test]
    fn get_size_work() {
        let m = Matrix::<i32>::new(43, 674, 0);

        assert_eq!(m.get_size_i(), 43);
        assert_eq!(m.get_size_j(), 674);
    }
}
