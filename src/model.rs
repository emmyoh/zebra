use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use std::error::Error;

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
    fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        documents: Vec<S>,
    ) -> Result<Vec<Embedding>, Box<dyn Error>>;
}

impl DatabaseEmbeddingModel for TextEmbedding {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::BGESmallENV15,
            show_download_progress: false,
            ..Default::default()
        })?)
    }
    fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        documents: Vec<S>,
    ) -> Result<Vec<Embedding>, Box<dyn Error>> {
        Ok(self.embed(documents, None)?)
    }
}
