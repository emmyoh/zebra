use crate::{Embedding, EmbeddingPrecision};
use bitcode::{Decode, Encode};
use simsimd::SpatialSimilarity;

#[derive(Encode, Decode)]
/// An `N`-dimensional hyperplane; a hyperplane is a generalisation of a line (which has one dimension) or plane (which has two dimensions).
///
/// It is defined when the dot product of a normal vector and some other vector, plus a constant, equals zero.
pub struct Hyperplane<const N: usize> {
    /// The vector normal to the hyperplane.
    pub coefficients: Embedding<N>,
    /// The offset of the hyperplane.
    pub constant: EmbeddingPrecision,
}

impl<const N: usize> Hyperplane<N> {
    /// Calculates if a point is 'above' the hyperplane.
    ///
    /// A point is 'above' a hyperplane when it is pointing in the same direction as the hyperplane's normal vector.
    ///
    /// # Arguments
    ///
    /// * `point` - The point which may be above, on, or below the hyperplane.
    ///
    /// # Returns
    ///
    /// If the given point is above the hyperplane.
    pub fn point_is_above(&self, point: &Embedding<N>) -> bool {
        EmbeddingPrecision::dot(&self.coefficients, point).unwrap_or_default()
            + self.constant as f64
            >= 0.0
    }
}
