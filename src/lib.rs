#![doc = include_str!("../README.md")]
// #![feature(doc_auto_cfg)]
#![warn(missing_docs)]

/// A module for database operations regardless of data type.
pub mod database;
/// A module for distance metrics.
pub mod distance;
/// A module for embedding models.
pub mod model;

/// An embedding vector.
pub type Embedding<const N: usize> = [EmbeddingPrecision; N];
/// The floating-point precision used to represent embedding vectors.
pub type EmbeddingPrecision = f32;
static KEYSPACE: std::sync::LazyLock<fjall::Keyspace> = std::sync::LazyLock::new(|| {
    fjall::Config::new("keyspace")
        .open()
        .expect("Keyspace should be accessible from disk")
});
