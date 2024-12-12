#![doc = include_str!("../README.md")]
#![feature(doc_auto_cfg)]
#![warn(missing_docs)]

/// A module for audio database operations.
pub mod audio;
/// A module for database operations regardless of data type.
pub mod db;
/// A module for distance metrics.
pub mod distance;
/// A module for image database operations.
pub mod image;
/// A module for embedding models.
pub mod model;
/// A module for text database operations.
pub mod text;
