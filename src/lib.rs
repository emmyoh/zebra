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

/// The candidate list size for the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Can be changed after database creation.
pub const EF: usize = 24;
