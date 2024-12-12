use crate::db::{Database, DocumentType};
use crate::distance::{CosineDistance, DefaultImageMetric};
use bytes::Bytes;
use candle_core::Tensor;
use candle_examples::imagenet::{IMAGENET_MEAN, IMAGENET_STD};
use image::ImageReader;
use std::error::Error;
use std::io::Cursor;

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const IMAGE_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const IMAGE_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const IMAGE_M0: usize = 24;

/// A database containing images and their embeddings.
pub type ImageDatabase = Database<DefaultImageMetric, IMAGE_EF_CONSTRUCTION, IMAGE_M, IMAGE_M0>;

/// Load the image database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A database containing a HNSW graph and the inserted images.
pub fn create_or_load_database() -> Result<ImageDatabase, Box<dyn std::error::Error>> {
    ImageDatabase::create_or_load_database(CosineDistance, DocumentType::Image)
}

/// Loads an image from raw bytes with ImageNet normalisation applied, returning a tensor with the shape [3 224 224].
///
/// # Arguments
///
/// * `bytes` - The raw bytes of an image.
///
/// # Returns
///
/// A tensor with the shape [3 224 224]; ImageNet normalisation is applied.
pub fn load_image224(bytes: Bytes) -> Result<Tensor, Box<dyn Error>> {
    let res = 224_usize;
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let data = img.into_raw();
    let data =
        Tensor::from_vec(data, (res, res, 3), &candle_core::Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&IMAGENET_MEAN, &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&IMAGENET_STD, &candle_core::Device::Cpu)?.reshape((3, 1, 1))?;
    Ok((data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}
