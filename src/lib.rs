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
pub type Embedding = Vec<f32>;
