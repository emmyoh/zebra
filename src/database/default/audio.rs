use crate::{database::core::Database, distance::CosineDistance, model::audio::VitBasePatch16_224};

/// The default distance metric for audio embeddings.
pub type DefaultAudioMetric = CosineDistance;

/// The default embedding model for audio embeddings.
pub type DefaultAudioModel = VitBasePatch16_224;

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
pub const DEFAULT_AUDIO_EF_CONSTRUCTION: usize = 400;

/// The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
pub const DEFAULT_AUDIO_M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub const DEFAULT_AUDIO_M0: usize = 24;

/// A database containing sounds and their embeddings.
pub type DefaultAudioDatabase =
    Database<DefaultAudioMetric, DEFAULT_AUDIO_EF_CONSTRUCTION, DEFAULT_AUDIO_M, DEFAULT_AUDIO_M0>;

/// Load the audio database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A vector database for audio.
pub fn create_or_load_database() -> Result<DefaultAudioDatabase, Box<dyn std::error::Error>> {
    DefaultAudioDatabase::create_or_load_database(
        DefaultAudioMetric {},
        DefaultAudioModel {}.into(),
    )
}
