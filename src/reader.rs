use std::{
    fs,
    fmt::{Debug, Formatter, Error as FmtError},
};

#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    value: Box<[u8; 784]>,
}

impl Image {
    pub fn new(value: Box<[u8; 784]>) -> Self {
        Self {
            value,
        }
    }
}

impl Debug for Image {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        for i in 0..28 {
            let mut line = "".to_string();
            for j in 0..28 {
                line.push(match self.value[i * 28 + j] > 200 {
                    true => '#',
                    false => ' ',
                });
            }
            write!(f, "{line}\n");
        }
        Ok(())
    }
}

pub fn get_train_images() -> Vec<Image> {
    let mut file = fs::read("../mnist/train-images/train-images.idx").unwrap();
    for _ in 0..16 {
        file.remove(0);
    }
    let mut to_return: Vec<Image> = vec![];
    // for i in file 
    {
        let mut counter = 0;
        let mut value = Box::new([0; 784]);
        while counter < 784 {
            eprintln!("{}", file[counter]);
            value[counter] = file[counter];
            counter += 1;
        }
        to_return.push(Image::new(value));
    }
    to_return
}
