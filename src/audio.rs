use crate::db::{Database, DocumentType};
use crate::distance::{CosineDistance, DefaultAudioMetric};

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const AUDIO_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const AUDIO_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const AUDIO_M0: usize = 24;

/// A database containing sounds and their embeddings.
pub type AudioDatabase = Database<DefaultAudioMetric, AUDIO_EF_CONSTRUCTION, AUDIO_M, AUDIO_M0>;

/// Load the audio database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A vector database for audio.
pub fn create_or_load_database() -> Result<AudioDatabase, Box<dyn std::error::Error>> {
    AudioDatabase::create_or_load_database(CosineDistance, DocumentType::Audio)
}
