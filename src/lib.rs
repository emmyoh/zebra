#![doc = include_str!("../README.md")]
// #![feature(doc_auto_cfg)]
#![warn(missing_docs)]

/// Implementation of database operations.
pub mod database;
/// Implementation of distance metrics.
pub mod distance;
/// Interface for embedding models.
pub mod model;

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// An embedding vector.
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct Embedding<const N: usize>(#[serde_as(as = "[_; N]")] [EmbeddingPrecision; N]);
impl<const N: usize> std::ops::Deref for Embedding<N> {
    type Target = [EmbeddingPrecision; N];

    fn deref(&self) -> &[EmbeddingPrecision; N] {
        &self.0
    }
}
impl<const N: usize> std::ops::DerefMut for Embedding<N> {
    fn deref_mut(&mut self) -> &mut [EmbeddingPrecision; N] {
        &mut self.0
    }
}
impl<const N: usize> Default for Embedding<N> {
    fn default() -> Self {
        Self([0.0; N])
    }
}
impl<const N: usize> From<[EmbeddingPrecision; N]> for Embedding<N> {
    fn from(value: [EmbeddingPrecision; N]) -> Self {
        Self(value)
    }
}
impl<const N: usize> TryFrom<Vec<EmbeddingPrecision>> for Embedding<N> {
    type Error = Vec<EmbeddingPrecision>;
    fn try_from(value: Vec<EmbeddingPrecision>) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}
/// The floating-point precision used to represent embedding vectors.
pub type EmbeddingPrecision = f32;
static KEYSPACE: std::sync::LazyLock<fjall::Keyspace> = std::sync::LazyLock::new(|| {
    fjall::Config::new("keyspace")
        .open()
        .expect("Keyspace should be accessible from disk")
});
