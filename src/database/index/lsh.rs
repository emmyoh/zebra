use std::ops::Deref;

use crate::{distance::DistanceUnit, Embedding, EmbeddingPrecision, KEYSPACE};
use dashmap::DashSet;
use fjall::{KvSeparationOptions, PartitionCreateOptions, PartitionHandle, PersistMode};
use rand::seq::IteratorRandom;
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use space::Metric;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        EmbeddingPrecision::dot(self.coefficients.deref(), point.deref()).unwrap_or_default()
            + self.constant as f64
            >= 0.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InnerNode<const N: usize> {
    hyperplane: Hyperplane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeafNode(Vec<Uuid>);

#[derive(Clone)]
struct KeyValue {
    _keyspace: fjall::Keyspace,
    trees: PartitionHandle,
    embeddings: PartitionHandle,
}

impl KeyValue {
    fn new(uuid: &Uuid) -> anyhow::Result<Self> {
        let keyspace = fjall::Config::new(format!("{}-keyspace", uuid.as_simple())).open()?;
        let trees = keyspace.open_partition(
            &format!("{}-trees", uuid.as_simple()),
            PartitionCreateOptions::default().with_kv_separation(KvSeparationOptions::default()),
        )?;
        let embeddings = keyspace.open_partition(
            &format!("{}-embeddings", uuid.as_simple()),
            PartitionCreateOptions::default().with_kv_separation(KvSeparationOptions::default()),
        )?;
        Ok(Self {
            _keyspace: keyspace,
            trees,
            embeddings,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Creation options for an [`LSHIndex`].
pub struct LSHIndexOptions<const N: usize> {
    /// The maximum number of vectors allowed on one side of a hyperplane; as the maximum node size decreases, the number of partitions in the database grows, increasing query accuracy ('recall') but decreasing performance.
    pub max_node_size: usize,
}

impl<const N: usize> Default for LSHIndexOptions<N> {
    fn default() -> Self {
        Self { max_node_size: 15 }
    }
}

/// An implementation of [the random projection method of locality sensitive hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection) as a data structure for use as a database index.
///
/// This index stores vectors on disk, minimising memory usage.
/// Memory-mapped file IO [is *not* used](https://db.cs.cmu.edu/mmap-cidr2022/).
#[derive(Clone)]
pub struct LSHIndex<const N: usize> {
    max_node_size: usize,
    kv: KeyValue,
}

impl<const N: usize> LSHIndex<N> {
    /// Construct a new [LSHIndex].
    ///
    /// # Arguments
    ///
    /// * `uuid` - The UUID of the database.
    ///
    /// * `options` - The creation options for the index.
    ///
    /// # Returns
    ///
    /// An [LSHIndex].
    pub fn new(uuid: &Uuid, options: &LSHIndexOptions<N>) -> anyhow::Result<Self> {
        Ok(Self {
            max_node_size: options.max_node_size,
            kv: KeyValue::new(uuid)?,
        })
    }

    fn vector(&self, embeddings: &PartitionHandle, idx: &Uuid) -> Embedding<N> {
        bincode::serde::decode_from_slice(
            &embeddings
                .get(idx.as_bytes())
                .ok()
                .flatten()
                .unwrap_or(fjall::Slice::from(vec![])),
            bincode::config::legacy(),
        )
        .unwrap_or_default()
        .0
    }

    fn subtract(lhs: &Embedding<N>, rhs: &Embedding<N>) -> Embedding<N> {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_default()
    }

    fn average(lhs: &Embedding<N>, rhs: &Embedding<N>) -> Embedding<N> {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_default()
    }

    fn build_hyperplane(
        &self,
        indexes: &Vec<Uuid>,
    ) -> anyhow::Result<(Hyperplane<N>, Vec<Uuid>, Vec<Uuid>)> {
        // Pick two random vectors
        let samples: Vec<_> = self
            .kv
            .embeddings
            .iter()
            .choose_multiple(&mut rand::rng(), 2);

        let empty_slice = fjall::Slice::from(vec![]);
        let a = samples
            .first()
            .and_then(|x| x.as_ref().map(|y| &y.1).ok())
            .unwrap_or(&empty_slice);
        let b = samples
            .get(1)
            .and_then(|x| x.as_ref().map(|y| &y.1).ok())
            .unwrap_or(&empty_slice);
        let (a, b) = (
            bincode::serde::decode_from_slice(a, bincode::config::legacy())
                .unwrap_or_default()
                .0,
            bincode::serde::decode_from_slice(b, bincode::config::legacy())
                .unwrap_or_default()
                .0,
        );

        // Use the two random points to make a hyperplane orthogonal to a line connecting the two points
        let coefficients = Self::subtract(&b, &a);
        let point_on_plane = Self::average(&a, &b);
        let constant = -EmbeddingPrecision::dot(coefficients.deref(), point_on_plane.deref())
            .unwrap_or_default() as EmbeddingPrecision;

        let hyperplane = Hyperplane {
            coefficients,
            constant,
        };

        // For each given vector ID, classify a vector as above or below the hyperplane.
        let above: DashSet<Uuid> = DashSet::new();
        let below: DashSet<Uuid> = DashSet::new();

        indexes.into_par_iter().for_each(|id| {
            match hyperplane.point_is_above(&self.vector(&self.kv.embeddings, id)) {
                true => above.insert(*id),
                false => below.insert(*id),
            };
        });

        Ok((
            hyperplane,
            above.into_par_iter().collect(),
            below.into_par_iter().collect(),
        ))
    }

    fn build_a_tree(&self, indexes: &Vec<Uuid>) -> anyhow::Result<Node<N>> {
        match indexes.len() < self.max_node_size {
            true => Ok(Node::Leaf(Box::new(LeafNode(indexes.clone())))),
            false => {
                // If there are too many indices to fit into a leaf node, recursively build trees to spread the indices across leaf nodes.
                let (hyperplane, above, below) = self.build_hyperplane(indexes)?;

                let node_above = self.build_a_tree(&above)?;
                let node_below = self.build_a_tree(&below)?;

                Ok(Node::Inner(Box::new(InnerNode {
                    hyperplane,
                    left_node: node_below,
                    right_node: node_above,
                })))
            }
        }
    }

    /// Remove duplicate embedding vectors from the index.
    pub fn deduplicate(&self) -> anyhow::Result<DashSet<Uuid>> {
        let seen: DashSet<Vec<_>> = DashSet::new();
        let mut to_remove = Vec::new();
        for kv in self.kv.embeddings.iter() {
            let (k, v) = kv?;
            let id = Uuid::from_slice(&k)?;
            let embedding: Embedding<N> =
                bincode::serde::decode_from_slice(&v, bincode::config::legacy())?.0;
            // The embedding itself cannot be hashed, so we look at its bits
            let embedding_bits: Vec<_> = embedding.iter().map(|x| x.to_bits()).collect();
            match seen.contains(&embedding_bits) {
                true => to_remove.push(id),
                false => {
                    seen.insert(embedding_bits);
                }
            }
        }
        self.remove(&to_remove)
    }

    fn tree_result<Met: Metric<Embedding<N>, Unit = DistanceUnit> + Send + Sync>(
        &self,
        query: &Embedding<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &DashSet<Uuid>,
        metric: &Met,
    ) -> anyhow::Result<i32> {
        match tree {
            Node::Leaf(leaf_node) => {
                let leaf_values_index = &(leaf_node.0);
                match leaf_values_index.len() < n as usize {
                    true => {
                        // If there are less than `n` vectors in this leaf node, they're all part of the candidate list.
                        leaf_values_index.into_par_iter().for_each(|i| {
                            candidates.insert(*i);
                        });
                        Ok(leaf_values_index.len() as i32)
                    }
                    false => {
                        let mut sorted_candidates = leaf_values_index
                            .into_par_iter()
                            .map(|idx| {
                                let curr_vector: Embedding<N> =
                                    self.vector(&self.kv.embeddings, idx);
                                (idx, metric.distance(&curr_vector, query))
                            })
                            .collect::<Vec<_>>();
                        sorted_candidates
                            .par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let top_candidates: Vec<Uuid> = sorted_candidates
                            .iter()
                            .take(n as usize)
                            .map(|(idx, _)| **idx)
                            .collect();

                        top_candidates.into_par_iter().for_each(|i| {
                            candidates.insert(i);
                        });

                        Ok(n)
                    }
                }
            }
            Node::Inner(inner_node) => {
                let is_above = inner_node.hyperplane.point_is_above(query);
                let (main, backup) = match is_above {
                    true => (&inner_node.right_node, &inner_node.left_node),
                    false => (&inner_node.left_node, &inner_node.right_node),
                };

                Ok(
                    match self.tree_result(query, n, main, candidates, metric)? {
                        k if k < n => self.tree_result(query, n - k, backup, candidates, metric)?,
                        k => k,
                    },
                )
            }
        }
    }

    fn insert(
        &self,
        current_node: &mut Node<N>,
        embedding: &Embedding<N>,
        vec_id: Uuid,
    ) -> anyhow::Result<()> {
        match current_node {
            Node::Inner(inner_node) => {
                let is_above = inner_node.hyperplane.point_is_above(embedding);

                let next_node = match is_above {
                    true => &mut inner_node.right_node,
                    false => &mut inner_node.left_node,
                };

                self.insert(next_node, embedding, vec_id)?;
            }
            Node::Leaf(leaf_node) => {
                match leaf_node.0.len() + 1 > self.max_node_size {
                    false => leaf_node.0.push(vec_id),
                    true => {
                        // If adding the vector ID to this leaf node would cause it to be too large, split this node.
                        let mut new_indexes = leaf_node.0.clone();
                        new_indexes.push(vec_id);

                        let result_node = self.build_a_tree(&new_indexes)?;
                        *current_node = result_node;
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether or not the index is empty.
    ///
    /// # Returns
    ///
    /// If the index is empty. Note that this is not the same as having no vectors in the index, as deleting all vectors in existing trees does not delete the trees themselves.
    pub fn is_empty(&self) -> bool {
        self.no_vectors() || self.no_trees()
    }

    /// Whether or not there are no vectors in the index.
    ///
    /// # Returns
    ///
    /// If there are no vectors in the index.
    pub fn no_vectors(&self) -> bool {
        self.kv.embeddings.is_empty().unwrap_or(true)
    }

    /// Whether or not there are no trees in the index.
    ///
    /// # Returns
    ///
    /// If there are no hyperplanes partitioning the vectors in the index.
    pub fn no_trees(&self) -> bool {
        self.kv.trees.is_empty().unwrap_or(true)
    }

    /// Adds vectors to the index.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - The embedding vectors to add to the index.
    ///
    /// # Returns
    ///
    /// A list of the IDs for the added embedding vectors.
    pub fn add(&self, embeddings: &Vec<Embedding<N>>) -> anyhow::Result<Vec<Uuid>> {
        let vector_ids = embeddings
            .par_iter()
            .map(|embedding| -> anyhow::Result<Uuid> {
                let vec_id = Uuid::now_v7();
                self.kv.embeddings.insert(
                    vec_id.as_bytes(),
                    bincode::serde::encode_to_vec(embedding, bincode::config::legacy())?,
                )?;

                for kv in self.kv.trees.iter() {
                    let (k, v) = kv?;
                    let id = Uuid::from_slice(&k)?;
                    let mut tree: Node<N> =
                        bincode::serde::decode_from_slice(&v, bincode::config::legacy())?.0;
                    self.insert(&mut tree, embedding, vec_id)?;
                    self.kv.trees.insert(
                        id.as_bytes(),
                        bincode::serde::encode_to_vec(&tree, bincode::config::legacy())?,
                    )?;
                }

                Ok(vec_id)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        KEYSPACE.persist(PersistMode::SyncAll)?;
        Ok(vector_ids)
    }

    /// Removes vectors from the index.
    ///
    /// # Arguments
    ///
    /// * `embedding_ids` - The IDs of the vectors to remove.
    pub fn remove(&self, embedding_ids: &Vec<Uuid>) -> anyhow::Result<DashSet<Uuid>> {
        let removed = DashSet::new();

        embedding_ids
            .par_iter()
            .map(|x| -> anyhow::Result<()> {
                self.kv
                    .trees
                    .iter()
                    .try_for_each(|kv| -> anyhow::Result<()> {
                        let (k, v) = kv?;
                        let id = Uuid::from_slice(&k)?;
                        let mut tree: Node<N> =
                            bincode::serde::decode_from_slice(&v, bincode::config::legacy())?.0;
                        if let Node::Leaf(leaf) = tree {
                            let leaf_nodes: Vec<Uuid> =
                                (*leaf).0.into_par_iter().filter(|y| y != x).collect();
                            tree = Node::Leaf(Box::new(LeafNode(leaf_nodes)));
                            self.kv.trees.insert(
                                id.as_bytes(),
                                bincode::serde::encode_to_vec(&tree, bincode::config::legacy())?,
                            )?;
                        }
                        Ok(())
                    })?;
                self.kv.embeddings.remove(x.as_bytes())?;
                removed.insert(*x);
                Ok(())
            })
            .collect::<anyhow::Result<()>>()?;

        KEYSPACE.persist(PersistMode::SyncAll)?;
        Ok(removed)
    }

    /// Delete the contents of the index.
    pub fn clear(&self) -> anyhow::Result<()> {
        self.kv
            .embeddings
            .iter()
            .map(|kv| -> anyhow::Result<()> {
                let (k, _) = kv?;
                self.kv.embeddings.remove(k)?;
                Ok(())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.kv
            .trees
            .iter()
            .map(|kv| -> anyhow::Result<()> {
                let (k, _) = kv?;
                self.kv.embeddings.remove(k)?;
                Ok(())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        KEYSPACE.persist(PersistMode::SyncAll)?;
        Ok(())
    }

    /// Perform an approximate *k* nearest neighbours search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector.
    ///
    /// * `top_k` - The number of approximate neighbours to the query vector to return.
    ///
    /// * `metric` - The distance metric used to evaluate the distances between the vectors in the index and the query vector.
    ///
    /// # Returns
    ///
    /// The IDs of, and distances from, `top_k` approximate nearest neighbours of the query vector.
    pub fn search<Met: Metric<Embedding<N>, Unit = DistanceUnit> + Send + Sync>(
        &self,
        query: &Embedding<N>,
        top_k: usize,
        metric: &Met,
    ) -> anyhow::Result<Vec<(Uuid, DistanceUnit)>> {
        let candidates = DashSet::new();

        for kv in self.kv.trees.iter() {
            let (_, v) = kv?;
            let tree: Node<N> = bincode::serde::decode_from_slice(&v, bincode::config::legacy())?.0;
            self.tree_result(query, top_k as i32, &tree, &candidates, metric)?;
        }

        let mut sorted_candidates = candidates
            .into_par_iter()
            .map(|idx| {
                (
                    idx,
                    metric.distance(&self.vector(&self.kv.embeddings, &idx), query),
                )
            })
            .collect::<Vec<_>>();
        sorted_candidates.par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(sorted_candidates.into_iter().take(top_k).collect())
    }
}
