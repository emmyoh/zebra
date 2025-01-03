use crate::Embedding;
use bytes::Bytes;
use serde::Serialize;

/// Dimensionality of embeddings produced by the [crate::model::text::BGESmallEn1_5] model.
pub const DIM_BGESMALL_EN_1_5: usize = 384;

/// Dimensionality of embeddings produced by the [crate::model::image::VitBasePatch16_224] model.
pub const DIM_VIT_BASE_PATCH16_224: usize = 768;

/// A trait for embedding models that can be used with the database.
pub trait DatabaseEmbeddingModel<const N: usize>: Serialize {
    /// Embed a vector of documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be embedded.
    ///
    /// # Returns
    ///
    /// A vector of embeddings.
    fn embed_documents(&self, documents: &[Bytes]) -> anyhow::Result<Vec<Embedding<N>>>;

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
        self.embed_documents(&[document])
            .map(|x| x.into_iter().next().unwrap_or_default())
    }
}
