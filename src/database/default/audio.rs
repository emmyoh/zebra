use crate::{
    database::core::Database,
    distance::CosineDistance,
    model::{audio::VitBasePatch16_224, core::DIM_VIT_BASE_PATCH16_224},
};

/// The default distance metric for audio embeddings.
pub type DefaultAudioMetric = CosineDistance<DIM_VIT_BASE_PATCH16_224>;

/// The default embedding model for audio embeddings.
pub type DefaultAudioModel = VitBasePatch16_224;

/// A database containing sounds and their embeddings.
pub type DefaultAudioDatabase =
    Database<DIM_VIT_BASE_PATCH16_224, DefaultAudioMetric, DefaultAudioModel>;
