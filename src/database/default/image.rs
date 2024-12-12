use crate::database::core::Database;
use crate::distance::CosineDistance;
use crate::model::image::VitBasePatch16_224;

/// The default distance metric for image embeddings.
pub type DefaultImageMetric = CosineDistance;

/// The default embedding model for image embeddings.
pub type DefaultImageModel = VitBasePatch16_224;

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const DEFAULT_IMAGE_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const DEFAULT_IMAGE_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const DEFAULT_IMAGE_M0: usize = 24;

/// A database containing images and their embeddings.
pub type DefaultImageDatabase = Database<
    DefaultImageMetric,
    DefaultImageModel,
    DEFAULT_IMAGE_EF_CONSTRUCTION,
    DEFAULT_IMAGE_M,
    DEFAULT_IMAGE_M0,
>;

/// Load the image database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A vector database for images.
pub fn create_or_load_database() -> Result<DefaultImageDatabase, Box<dyn std::error::Error>> {
    DefaultImageDatabase::create_or_load_database(DefaultImageMetric {}, DefaultImageModel {})
}
