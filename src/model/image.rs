use super::core::DatabaseEmbeddingModel;
use crate::database::core::DocumentType;
use crate::Embedding;
use bytes::Bytes;
use candle_core::{DType, Tensor};
use candle_examples::imagenet::{IMAGENET_MEAN, IMAGENET_STD};
use candle_nn::VarBuilder;
use candle_transformers::models::vit;
use image::ImageReader;
use serde::{Deserialize, Serialize};
use std::{error::Error, io::Cursor};

/// A trait for image embedding models.
pub trait ImageEmbeddingModel {
    /// Loads an image from raw bytes with ImageNet normalisation applied, returning a tensor with the shape [3 224 224].
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes of an image.
    ///
    /// # Returns
    ///
    /// A tensor with the shape [3 224 224]; ImageNet normalisation is applied.
    fn load_image224(&self, bytes: Bytes) -> Result<Tensor, Box<dyn Error>> {
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
}

/// A model for embedding images.
#[derive(Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VitBasePatch16_224;
impl ImageEmbeddingModel for VitBasePatch16_224 {}

#[typetag::serde]
impl DatabaseEmbeddingModel for VitBasePatch16_224 {
    fn document_type(&self) -> DocumentType {
        DocumentType::Image
    }
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>> {
        let mut result = Vec::new();
        let device = candle_examples::device(false)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(
            &vit::Config::vit_base_patch16_224(),
            false,
            varbuilder.pp("vit").pp("embeddings"),
        )?;
        for document in documents {
            let image = self.load_image224(document)?.to_device(&device)?;
            let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
            let embedding_vector = embedding_tensors.flatten_all()?.to_vec1::<f32>()?;
            result.push(embedding_vector);
        }
        Ok(result)
    }
    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>> {
        let device = candle_examples::device(false)?;
        let image = self.load_image224(document)?.to_device(&device)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(&vit::Config::vit_base_patch16_224(), false, varbuilder)?;
        let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
        let embedding_vector = embedding_tensors.to_vec1::<f32>()?;
        Ok(embedding_vector)
    }
}
