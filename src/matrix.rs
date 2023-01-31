use std::fmt::{Display, Debug, Formatter};
use std::fmt;
use std::error::Error as StdError;
use std::alloc::{alloc, dealloc, Layout};
use std::ops::Drop;
use std::mem::{size_of, align_of};

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

        let to_return = Self::get_new(size_i, size_j);
        
        for i in 0..to_return.size_i * to_return.size_j {
            unsafe {
                *to_return.get_ptr_to(i) = value.clone();
            }
        }
        to_return
    }

    /// Create a new `Matrix` filled with default value
    pub fn new_default(size_i: usize, size_j: usize) -> Self where
    T: Default {
        let to_return = Self::get_new(size_i, size_j);

        for i in 0..to_return.size_i * to_return.size_j {
            unsafe {
                *to_return.get_ptr_to(i) = T::default();
            }
        }

        to_return
    }

    /// Create a new identity `Matrix`
    pub fn new_id(size: usize) -> Self where
    T: Clone + Zero + One {
        let to_return = Self::get_new(size, size);

        for i in 0..to_return.size_i {
            for j in 0..to_return.size_j {
                unsafe {
                    *to_return.get_ptr_to(to_return.convert_index(i, j)) = if i == j { T::one() } else { T::zero() };
                }
            }
        }

        to_return
    }
    
    fn get_new(size_i: usize, size_j: usize) -> Self {
        Self {
            value: Self::allocate(size_i, size_j),
            size_i: size_i,
            size_j: size_j,
        }
    }

    fn allocate(size_i: usize, size_j: usize) -> *mut T {
        let layout = unsafe { Layout::from_size_align_unchecked(size_of::<T>() * size_i * size_j, align_of::<T>()) };
        unsafe {
            alloc(layout) as *mut T
        }
    }

    fn get_ptr_to(&self, index: usize) -> *mut T {
        (self.value as usize + (index * size_of::<T>())) as *mut T
    }
    
    fn convert_index(&self, i: usize, j: usize) -> usize {
        i * self.size_j + j
    }
}

/// # Getters and Setters for `Matrix<T>`
impl<T> Matrix<T> {
    /// Return a reference to the value at index i, j of `self`
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if self.check_index(i, j) {
            unsafe {
                Some(&*self.get_ptr_to(self.convert_index(i, j)))
            }
        } else {
            None
        }
    }

    /// Return a mutable reference to the value at index i, j of `self`
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if self.check_index(i, j) {
            unsafe {
                Some(&mut *self.get_ptr_to(self.convert_index(i, j)))
            }
        } else {
            None
        }
    }

    /// Return the I size of `self`
    pub fn get_size_i(&self) -> usize {
        self.size_i
    }

    /// Return the J size of `self`
    pub fn get_size_j(&self) -> usize {
        self.size_j
    }

    /// Return true if there is a value at the index i, j in `self` and false if the index is out
    /// of range
    pub fn check_index(&self, i: usize, j: usize) -> bool {
        i < self.size_i && j < self.size_j
    }

}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.value as *mut u8, Layout::from_size_align_unchecked(size_of::<T>(), align_of::<T>()));
        }
        drop(self);
    }
}

/// An enum representing an error
#[derive(Debug)]
pub enum Error {
    Addition,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let msg = String::from(
            match self {
                Self::Addition => "You cannot add these two matrices",
            }
        );

        write!(f, "{}", msg)
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

    #[test]
    fn get_work() {
        let mut m = Matrix::<i32>::new_default(24, 54);

        assert_eq!(m.get(24, 3), None);
        assert_eq!(m.get_mut(3, 54), None);
    }
}
