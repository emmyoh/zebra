use super::core::{DatabaseEmbeddingModel, DIM_BGESMALL_EN_1_5};
use crate::Embedding;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// A model for embedding text.
#[derive(
    Default, Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub struct BGESmallEn1_5;

impl DatabaseEmbeddingModel<DIM_BGESMALL_EN_1_5> for BGESmallEn1_5 {
    fn embed_documents(&self, documents: &[bytes::Bytes]) -> anyhow::Result<Vec<Embedding<384>>> {
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
            .map(|x| x.try_into().unwrap_or_default())
            .collect())
    }
}
