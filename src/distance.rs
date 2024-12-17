use crate::Embedding;
use distances::vectors::{
    bray_curtis, canberra, chebyshev, cosine, euclidean, hamming, l3_norm, l4_norm, manhattan,
    minkowski, minkowski_p,
};
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use space::Metric;

/// The data type representing the distance between two embeddings.
pub type DistanceUnit = u64;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The cosine distance metric.
pub struct CosineDistance;

impl Metric<Embedding> for CosineDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let cosine_distance: f32 = cosine(a, b);
        cosine_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L2-squared distance metric.
pub struct L2SquaredDistance;

impl Metric<Embedding> for L2SquaredDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        f32::sqeuclidean(a, b)
            .map(|x| x.to_bits())
            .unwrap_or_default()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Chebyshev distance metric.
pub struct ChebyshevDistance;

impl Metric<Embedding> for ChebyshevDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let chebyshev_distance = chebyshev(a, b);
        chebyshev_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Canberra distance metric.
pub struct CanberraDistance;

impl Metric<Embedding> for CanberraDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let canberra_distance: f32 = canberra(a, b);
        canberra_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Bray-Curtis distance metric.
pub struct BrayCurtisDistance;

impl Metric<Embedding> for BrayCurtisDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let bray_curtis_distance: f32 = bray_curtis(a, b);
        bray_curtis_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Manhattan distance metric.
pub struct ManhattanDistance;

impl Metric<Embedding> for ManhattanDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let manhattan_distance: f32 = manhattan(a, b);
        manhattan_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L2 distance metric.
pub struct L2Distance;

impl Metric<Embedding> for L2Distance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let l2_distance: f32 = euclidean(a, b);
        l2_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L3 distance metric.
pub struct L3Distance;

impl Metric<Embedding> for L3Distance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let l3_distance: f32 = l3_norm(a, b);
        l3_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L4 distance metric.
pub struct L4Distance;

impl Metric<Embedding> for L4Distance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let l4_distance: f32 = l4_norm(a, b);
        l4_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Hamming distance metric.
pub struct HammingDistance;

impl Metric<Embedding> for HammingDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let a_to_bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
        let b_to_bits: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
        let hamming_distance: u32 = hamming(a_to_bits.as_slice(), b_to_bits.as_slice());
        hamming_distance.into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Minkowski distance metric.
pub struct MinkowskiDistance {
    /// The power of the Minkowski distance.
    pub power: i32,
}

impl Metric<Embedding> for MinkowskiDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let metric = minkowski(self.power);
        let distance: f32 = metric(a, b);
        distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The p-norm distance metric.
pub struct PNormDistance {
    /// The power of the distance metric.
    pub power: i32,
}

impl Metric<Embedding> for PNormDistance {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let metric = minkowski_p(self.power);
        let distance: f32 = metric(a, b);
        distance.to_bits().into()
    }
}
