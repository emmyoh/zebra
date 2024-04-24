use crate::db::{Database, DocumentType};
use crate::distance::{CosineDistance, DefaultImageMetric};

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const IMAGE_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const IMAGE_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const IMAGE_M0: usize = 24;

/// A database containing images and their embeddings.
pub type ImageDatabase = Database<DefaultImageMetric, IMAGE_EF_CONSTRUCTION, IMAGE_M, IMAGE_M0>;

/// Load the image database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A database containing a HNSW graph and the inserted images.
pub fn create_or_load_database() -> Result<ImageDatabase, Box<dyn std::error::Error>> {
    ImageDatabase::create_or_load_database(CosineDistance, DocumentType::Image)
}
