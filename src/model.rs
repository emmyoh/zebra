use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::vit;
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use std::{error::Error, path::PathBuf};

/// A trait for embedding models that can be used with the database.
pub trait DatabaseEmbeddingModel {
    /// Create a new instance of the embedding model.
    fn new() -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    /// Embed a vector of documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be embedded.
    ///
    /// # Returns
    ///
    /// A vector of embeddings.
    fn embed_documents<S: AsRef<str> + Send + Sync>(
        &self,
        documents: Vec<S>,
    ) -> Result<Vec<Embedding>, Box<dyn Error>>;

    /// Embed a single document.
    ///
    /// # Arguments
    ///
    /// * `document` – A single document to be embedded.
    ///
    /// # Returns
    ///
    /// An embedding vector.
    fn embed<S: AsRef<str> + Send + Sync>(&self, document: S) -> Result<Embedding, Box<dyn Error>>;
}

impl DatabaseEmbeddingModel for TextEmbedding {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::BGESmallENV15,
            show_download_progress: false,
            ..Default::default()
        })?)
    }
    fn embed_documents<S: AsRef<str> + Send + Sync>(
        &self,
        documents: Vec<S>,
    ) -> Result<Vec<Embedding>, Box<dyn Error>> {
        Ok(self.embed(documents, None)?)
    }

    fn embed<S: AsRef<str> + Send + Sync>(&self, document: S) -> Result<Embedding, Box<dyn Error>> {
        let vec_with_document = vec![document];
        let vector_of_embeddings = self.embed(vec_with_document, None)?;
        Ok(vector_of_embeddings.first().unwrap().to_vec())
    }
}

/// A model for embedding images.
pub struct ImageEmbeddingModel;

impl DatabaseEmbeddingModel for ImageEmbeddingModel {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self)
    }
    fn embed_documents<S: AsRef<str> + Send + Sync>(
        &self,
        documents: Vec<S>,
    ) -> Result<Vec<Embedding>, Box<dyn Error>> {
        let mut result = Vec::new();
        let device = candle_examples::device(false)?;
        for document in documents {
            let path = PathBuf::from(document.as_ref().to_string());
            let image = candle_examples::imagenet::load_image224(path)?.to_device(&device)?;
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
            let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
            let embedding_vector = embedding_tensors.flatten_all()?.to_vec1::<f32>()?;
            result.push(embedding_vector);
        }
        Ok(result)
    }
    fn embed<S: AsRef<str> + Send + Sync>(&self, document: S) -> Result<Embedding, Box<dyn Error>> {
        let device = candle_examples::device(false)?;
        let path = PathBuf::from(document.as_ref().to_string());
        let image = candle_examples::imagenet::load_image224(path)?.to_device(&device)?;
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
