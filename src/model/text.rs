use super::core::DatabaseEmbeddingModel;
use crate::database::core::DocumentType;
use bytes::Bytes;
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::error::Error;

/// A model for embedding images.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BGESmallEn1_5;

impl DatabaseEmbeddingModel for BGESmallEn1_5 {
    fn document_type(&self) -> DocumentType {
        DocumentType::Text
    }
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )?;
        Ok(model.embed(
            documents
                .into_par_iter()
                .map(|x| x.to_vec())
                .filter_map(|x| String::from_utf8(x).ok())
                .collect(),
            None,
        )?)
    }

    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )?;
        let vec_with_document = vec![document]
            .into_par_iter()
            .map(|x| x.to_vec())
            .filter_map(|x| String::from_utf8(x).ok())
            .collect();
        let vector_of_embeddings = model.embed(vec_with_document, None)?;
        Ok(vector_of_embeddings.first().unwrap().to_vec())
    }
}
