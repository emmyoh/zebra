use super::core::{DatabaseEmbeddingModel, DIM_BGESMALL_EN_1_5};
use crate::Embedding;
use bitcode::{Decode, Encode};
use bytes::Bytes;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// A model for embedding text.
#[derive(Default, Encode, Decode, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BGESmallEn1_5;

impl DatabaseEmbeddingModel<DIM_BGESMALL_EN_1_5> for BGESmallEn1_5 {
    fn embed_documents(&self, documents: &Vec<Bytes>) -> anyhow::Result<Vec<Embedding<384>>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )?;
        let embeddings = model.embed(
            documents
                .into_par_iter()
                .map(|x| x.to_vec())
                .filter_map(|x| String::from_utf8(x).ok())
                .collect(),
            None,
        )?;
        Ok(embeddings
            .into_par_iter()
            .map(|x| x.try_into().unwrap_or([0.0; 384]))
            .collect())
    }
}
