use std::ops::Deref;

use crate::{Embedding, EmbeddingPrecision};
use distances::vectors::{
    bray_curtis, canberra, chebyshev, cosine, euclidean, euclidean_sq, hamming, l3_norm, l4_norm,
    manhattan, minkowski, minkowski_p,
};
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use space::Metric;

/// The data type representing the distance between two embeddings.
pub type DistanceUnit = u64;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The cosine distance metric.
pub struct CosineDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for CosineDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        // Use SIMD if vectors are of same length; otherwise, find distance after truncating longer vector so lengths match
        EmbeddingPrecision::cosine(a.deref(), b.deref())
            .map(|c| 1.0 - c)
            .map(|x| x.to_bits())
            .unwrap_or(
                cosine::<_, EmbeddingPrecision>(a.deref(), b.deref())
                    .to_bits()
                    .into(),
            )
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L2-squared distance metric.
pub struct L2SquaredDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for L2SquaredDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        EmbeddingPrecision::sqeuclidean(a.deref(), b.deref())
            .map(|x| x.to_bits())
            .unwrap_or(
                euclidean_sq::<_, EmbeddingPrecision>(a.deref(), b.deref())
                    .to_bits()
                    .into(),
            )
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Chebyshev distance metric.
pub struct ChebyshevDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for ChebyshevDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let chebyshev_distance = chebyshev(a.deref(), b.deref());
        chebyshev_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Canberra distance metric.
pub struct CanberraDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for CanberraDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let canberra_distance: EmbeddingPrecision = canberra(a.deref(), b.deref());
        canberra_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Bray-Curtis distance metric.
pub struct BrayCurtisDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for BrayCurtisDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let bray_curtis_distance: EmbeddingPrecision = bray_curtis(a.deref(), b.deref());
        bray_curtis_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Manhattan distance metric.
pub struct ManhattanDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for ManhattanDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let manhattan_distance: EmbeddingPrecision = manhattan(a.deref(), b.deref());
        manhattan_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L2 distance metric.
pub struct L2Distance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for L2Distance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        EmbeddingPrecision::euclidean(a.deref(), b.deref())
            .map(|x| x.to_bits())
            .unwrap_or(
                euclidean::<_, EmbeddingPrecision>(a.deref(), b.deref())
                    .to_bits()
                    .into(),
            )
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L3 distance metric.
pub struct L3Distance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for L3Distance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let l3_distance: EmbeddingPrecision = l3_norm(a.deref(), b.deref());
        l3_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The L4 distance metric.
pub struct L4Distance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for L4Distance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let l4_distance: EmbeddingPrecision = l4_norm(a.deref(), b.deref());
        l4_distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Hamming distance metric.
pub struct HammingDistance<const N: usize>;

impl<const N: usize> Metric<Embedding<N>> for HammingDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let a_to_bits: Vec<u8> = a.iter().map(|x| x.to_bits() as u8).collect();
        let b_to_bits: Vec<u8> = b.iter().map(|x| x.to_bits() as u8).collect();
        match a.len() == b.len() {
            true => hamming_bitwise_fast::hamming_bitwise_fast(
                a_to_bits.as_slice(),
                b_to_bits.as_slice(),
            )
            .into(),
            false => hamming::<_, u32>(a_to_bits.as_slice(), b_to_bits.as_slice()).into(),
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The Minkowski distance metric.
pub struct MinkowskiDistance<const N: usize> {
    /// The power of the Minkowski distance.
    pub power: i32,
}

impl<const N: usize> Metric<Embedding<N>> for MinkowskiDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let metric = minkowski(self.power);
        let distance: EmbeddingPrecision = metric(a.deref(), b.deref());
        distance.to_bits().into()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The p-norm distance metric.
pub struct PNormDistance<const N: usize> {
    /// The power of the distance metric.
    pub power: i32,
}

impl<const N: usize> Metric<Embedding<N>> for PNormDistance<N> {
    type Unit = DistanceUnit;
    fn distance(&self, a: &Embedding<N>, b: &Embedding<N>) -> Self::Unit {
        let metric = minkowski_p(self.power);
        let distance: EmbeddingPrecision = metric(a.deref(), b.deref());
        distance.to_bits().into()
    }
}
