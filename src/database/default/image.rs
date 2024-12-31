use crate::{
    database::core::Database,
    distance::CosineDistance,
    model::{core::DIM_VIT_BASE_PATCH16_224, image::VitBasePatch16_224},
};

/// The default distance metric for image embeddings.
pub type DefaultImageMetric = CosineDistance<DIM_VIT_BASE_PATCH16_224>;

/// The default embedding model for image embeddings.
pub type DefaultImageModel = VitBasePatch16_224;

/// A database containing images and their embeddings.
pub type DefaultImageDatabase =
    Database<DIM_VIT_BASE_PATCH16_224, DefaultImageMetric, DefaultImageModel>;
