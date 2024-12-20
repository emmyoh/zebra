use crate::database::core::DocumentType;
use crate::Embedding;
use bytes::Bytes;
use std::error::Error;

/// A trait for embedding models that can be used with the database.
#[typetag::serde(tag = "type")]
pub trait DatabaseEmbeddingModel {
    /// The type of document that can be embedded by this model.
    ///
    /// # Returns
    ///
    /// The document type supported by this database.
    fn document_type(&self) -> DocumentType;

    /// Embed a vector of documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be embedded.
    ///
    /// # Returns
    ///
    /// A vector of embeddings.
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>>;

    /// Embed a single document.
    ///
    /// # Arguments
    ///
    /// * `document` – A single document to be embedded.
    ///
    /// # Returns
    ///
    /// An embedding vector.
    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>>;
}

impl<'a, T: 'a> From<T> for Box<dyn DatabaseEmbeddingModel + 'a>
where
    T: DatabaseEmbeddingModel,
{
    fn from(v: T) -> Box<dyn DatabaseEmbeddingModel + 'a> {
        Box::new(v)
    }
}
