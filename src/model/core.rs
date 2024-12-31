use crate::Embedding;
use bitcode::{DecodeOwned, Encode};
use bytes::Bytes;
use dashmap::DashSet;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::error::Error;

pub const DIM_BGESMALL_EN_1_5: usize = 384;
pub const DIM_VIT_BASE_PATCH16_224: usize = 768;

/// A trait for embedding models that can be used with the database.
pub trait DatabaseEmbeddingModel<const N: usize>: Encode {
    /// Embed a vector of documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be embedded.
    ///
    /// # Returns
    ///
    /// A vector of embeddings.
    fn embed_documents(&self, documents: &Vec<Bytes>) -> anyhow::Result<Vec<Embedding<N>>>;

    /// Embed a single document.
    ///
    /// # Arguments
    ///
    /// * `document` – A single document to be embedded.
    ///
    /// # Returns
    ///
    /// An embedding vector.
    fn embed(&self, document: Bytes) -> anyhow::Result<Embedding<N>> {
        self.embed_documents(&vec![document])
            .map(|x| x.into_iter().next().unwrap_or([0.0; N]))
    }
}
