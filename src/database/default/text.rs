use crate::{
    database::core::Database,
    distance::L2SquaredDistance,
    model::{core::DIM_BGESMALL_EN_1_5, text::BGESmallEn1_5},
};

/// The default distance metric for text embeddings.
pub type DefaultTextMetric = L2SquaredDistance<DIM_BGESMALL_EN_1_5>;

/// The default embedding model for text embeddings.
pub type DefaultTextModel = BGESmallEn1_5;

/// A database containing texts and their embeddings.
pub type DefaultTextDatabase = Database<DIM_BGESMALL_EN_1_5, DefaultTextMetric, DefaultTextModel>;
