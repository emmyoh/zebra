use crate::db::{Database, DocumentType};
use crate::distance::{DefaultTextMetric, L2SquaredDistance};

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const TEXT_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const TEXT_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const TEXT_M0: usize = 24;

/// A database containing texts and their embeddings.
pub type TextDatabase = Database<DefaultTextMetric, TEXT_EF_CONSTRUCTION, TEXT_M, TEXT_M0>;

/// Load the text database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A database containing a HNSW graph and the inserted texts.
pub fn create_or_load_database() -> Result<TextDatabase, Box<dyn std::error::Error>> {
    TextDatabase::create_or_load_database(L2SquaredDistance, DocumentType::Text)
}
