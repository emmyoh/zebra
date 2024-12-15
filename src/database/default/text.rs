use crate::{database::core::Database, distance::L2SquaredDistance, model::text::BGESmallEn1_5};

/// The default distance metric for text embeddings.
pub type DefaultTextMetric = L2SquaredDistance;

/// The default embedding model for text embeddings.
pub type DefaultTextModel = BGESmallEn1_5;

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const DEFAULT_TEXT_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const DEFAULT_TEXT_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const DEFAULT_TEXT_M0: usize = 24;

/// A database containing texts and their embeddings.
pub type DefaultTextDatabase =
    Database<DefaultTextMetric, DEFAULT_TEXT_EF_CONSTRUCTION, DEFAULT_TEXT_M, DEFAULT_TEXT_M0>;

/// Load the text database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A vector database for text.
pub fn create_or_load_database() -> Result<DefaultTextDatabase, Box<dyn std::error::Error>> {
    DefaultTextDatabase::create_or_load_database(DefaultTextMetric {}, DefaultTextModel {}.into())
}
