use super::core::Hyperplane;
use crate::{distance::DistanceUnit, Embedding, EmbeddingPrecision, KEYSPACE};
use bitcode::{Decode, Encode};
use dashmap::DashSet;
use fjall::{KvSeparationOptions, PartitionCreateOptions, PartitionHandle, PersistMode};
use rand::seq::{IteratorRandom, SliceRandom};
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use simsimd::SpatialSimilarity;
use space::Metric;
use uuid::Uuid;

#[derive(Encode, Decode)]
enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode>),
}

#[derive(Encode, Decode)]
struct InnerNode<const N: usize> {
    hyperplane: Hyperplane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

#[derive(Encode, Decode)]
struct LeafNode(Vec<[u8; 16]>);

#[derive(Encode, Decode)]
pub struct ANNIndex<const N: usize> {
    uuid: [u8; 16],
    max_node_size: usize,
    // // vector containing all the trees that make up the index, with each element in the vector
    // // indicating the root node of the tree
    // trees: Vec<Node<N>>,
    // // stores all the vectors within the index in a global contiguous location
    // #[serde_as(as = "Vec<[_; N]>")]
    // values: Vec<Embedding<N>>,
    // ids: Vec<usize>,
}

impl<const N: usize> ANNIndex<N> {
    fn trees(&self) -> anyhow::Result<PartitionHandle> {
        Ok(KEYSPACE.open_partition(
            &format!("{}-trees", Uuid::from_bytes(self.uuid).as_simple()),
            PartitionCreateOptions::default().with_kv_separation(KvSeparationOptions::default()),
        )?)
    }

    fn embeddings(&self) -> anyhow::Result<PartitionHandle> {
        Ok(KEYSPACE.open_partition(
            &format!("{}-embeddings", Uuid::from_bytes(self.uuid).as_simple()),
            PartitionCreateOptions::default().with_kv_separation(KvSeparationOptions::default()),
        )?)
    }

    fn vector(&self, embeddings: &PartitionHandle, idx: &Uuid) -> Embedding<N> {
        bitcode::decode(
            &embeddings
                .get(bitcode::encode(idx.as_bytes()))
                .ok()
                .flatten()
                .unwrap_or(fjall::Slice::from(vec![])),
        )
        .unwrap_or([0.0; N])
    }

    fn tree(&self, trees: &PartitionHandle, id: &Uuid) -> anyhow::Result<Node<N>> {
        Ok(bitcode::decode(
            &trees
                .get(bitcode::encode(id.as_bytes()))
                .ok()
                .flatten()
                .unwrap_or(fjall::Slice::from(vec![])),
        )?)
    }

    fn subtract(lhs: &Embedding<N>, rhs: &Embedding<N>) -> Embedding<N> {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or([0.0; N])
    }

    fn average(lhs: &Embedding<N>, rhs: &Embedding<N>) -> Embedding<N> {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or([0.0; N])
    }

    fn build_hyperplane(
        &self,
        indexes: &Vec<[u8; 16]>,
    ) -> anyhow::Result<(Hyperplane<N>, Vec<[u8; 16]>, Vec<[u8; 16]>)> {
        let embeddings = self.embeddings()?;
        // sample two random vectors from the indexes, and use these two vectors to form a hyperplane boundary
        let samples: Vec<_> = embeddings
            .iter()
            .choose_multiple(&mut rand::thread_rng(), 2);

        let empty_slice = fjall::Slice::from(vec![]);
        let a = samples
            .get(0)
            .map(|x| x.as_ref().map(|y| &y.1).ok())
            .flatten()
            .unwrap_or(&empty_slice);
        let b = samples
            .get(1)
            .map(|x| x.as_ref().map(|y| &y.1).ok())
            .flatten()
            .unwrap_or(&empty_slice);
        let (a, b) = (
            bitcode::decode(&a).unwrap_or([0.0; N]),
            bitcode::decode(&b).unwrap_or([0.0; N]),
        );

        // construct the hyperplane by getting the plane's coefficients (normal vector), and the
        // its constant (the constant term from a plane's cartesian eqn).
        let coefficients = Self::subtract(&b, &a);
        let point_on_plane = Self::average(&a, &b);
        let constant =
            -EmbeddingPrecision::dot(&coefficients, &point_on_plane).unwrap_or_default() as f32;

        let hyperplane = Hyperplane {
            coefficients,
            constant,
        };

        // assign each index (which points to a specific vector in the embedding matrix) to
        // either `above` or `below` the hyperplane.
        let mut above: Vec<[u8; 16]> = Vec::new();
        let mut below: Vec<[u8; 16]> = Vec::new();

        for id in indexes {
            if hyperplane.point_is_above(&self.vector(&embeddings, Uuid::from_bytes_ref(id))) {
                above.push(*id);
            } else {
                below.push(*id);
            }
        }

        Ok((hyperplane, above, below))
    }

    fn build_a_tree(&self, indexes: &Vec<[u8; 16]>) -> anyhow::Result<Node<N>> {
        if indexes.len() < self.max_node_size {
            Ok(Node::Leaf(Box::new(LeafNode(indexes.clone()))))
        } else {
            let (hyperplane, above, below) = self.build_hyperplane(&indexes)?;

            let node_above = self.build_a_tree(&above)?;
            let node_below = self.build_a_tree(&below)?;

            Ok(Node::Inner(Box::new(InnerNode {
                hyperplane,
                left_node: node_below,
                right_node: node_above,
            })))
        }
    }

    // fn deduplicate(
    //     all_vectors: &Vec<Embedding<N>>,
    //     all_indexes: &Vec<usize>,
    //     dedup_vecs: &mut Vec<Embedding<N>>,
    //     dedup_vec_ids: &mut Vec<usize>,
    // ) {
    //     let mut hashes_seen: HashSet<HashKey<N>> = HashSet::new();
    //     for id in 0..all_vectors.len() {
    //         let hash_key = all_vectors[id].to_hashkey();
    //         if !hashes_seen.contains(&hash_key) {
    //             hashes_seen.insert(hash_key);

    //             // requires the Copy trait to copy a vector from `all_vectors` into `dedup_vecs`
    //             dedup_vecs.push(all_vectors[id]);
    //             dedup_vec_ids.push(all_indexes[id]);
    //         }
    //     }
    // }

    pub fn build_index(max_node_size: usize) -> Self {
        // let mut dedup_vecs = vec![];
        // let mut dedup_vec_ids = vec![];
        // // Self::deduplicate(&vectors, &vector_ids, &mut dedup_vecs, &mut dedup_vec_ids);

        // // maps each index to the unique vector
        // let all_indexes_from_unique_vecs = (0..dedup_vecs.len()).collect();

        // let trees: Vec<Node<N>> = (0..num_trees)
        //     .into_par_iter()
        //     .map(|_| Self::build_a_tree(max_size, &all_indexes_from_unique_vecs, &dedup_vecs))
        //     .collect();

        // println!(
        //     "Number of vectors in index: {}. Num of vec IDs in index: {}",
        //     dedup_vec_ids.len(),
        //     dedup_vecs.len()
        // );
        Self {
            uuid: Uuid::now_v7().into_bytes(),
            max_node_size,
            // trees,
            // values: dedup_vecs,
            // ids: dedup_vec_ids,
        }
    }

    fn tree_result<Met: Metric<Embedding<N>, Unit = DistanceUnit> + Send + Sync>(
        &self,
        query: Embedding<N>,
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
                        leaf_values_index.into_par_iter().for_each(|i| {
                            candidates.insert(Uuid::from_bytes(*i));
                        });
                        Ok(leaf_values_index.len() as i32)
                    }
                    false => {
                        let embeddings = self.embeddings()?;
                        let mut sorted_candidates = leaf_values_index
                            .into_par_iter()
                            .map(|idx| {
                                let curr_vector: Embedding<N> =
                                    self.vector(&embeddings, Uuid::from_bytes_ref(idx));
                                (idx, metric.distance(&curr_vector, &query))
                            })
                            .collect::<Vec<_>>();
                        sorted_candidates
                            .par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let top_candidates: Vec<Uuid> = sorted_candidates
                            .iter()
                            .take(n as usize)
                            .map(|(idx, _)| Uuid::from_bytes(**idx))
                            .collect();

                        top_candidates.into_par_iter().for_each(|i| {
                            candidates.insert(i);
                        });

                        // only take candidate whose vectors are those closest to the query in distance
                        Ok(n)
                    }
                }
            }
            Node::Inner(inner_node) => {
                let is_above = inner_node.hyperplane.point_is_above(&query);
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
                let is_above = inner_node.hyperplane.point_is_above(&embedding);

                let next_node = if is_above {
                    &mut inner_node.right_node
                } else {
                    &mut inner_node.left_node
                };

                self.insert(next_node, embedding, vec_id)?;
            }
            Node::Leaf(leaf_node) => {
                // split the node such that the current node becomes an InnerNode, if the number of elements exceed
                // the max node size. Otherwise, simply add the ID to the leaf node.
                if leaf_node.0.len() + 1 > self.max_node_size {
                    let mut new_indexes = leaf_node.0.clone();
                    new_indexes.push(vec_id.into_bytes());

                    let result_node = self.build_a_tree(&new_indexes)?;
                    *current_node = result_node;
                } else {
                    leaf_node.0.push(vec_id.into_bytes());
                }
            }
        }
        Ok(())
    }
}

impl<const N: usize> ANNIndex<N> {
    pub fn is_empty(&self) -> bool {
        self.embeddings()
            .ok()
            .map(|x| x.is_empty().ok())
            .flatten()
            .unwrap_or(true)
            || self
                .trees()
                .ok()
                .map(|x| x.is_empty().ok())
                .flatten()
                .unwrap_or(true)
    }

    pub fn add(&self, embeddings: &Vec<Embedding<N>>) -> anyhow::Result<Vec<Uuid>> {
        let trees = self.trees()?;
        let vectors = self.embeddings()?;

        let vector_ids = embeddings
            .par_iter()
            .map(|embedding| -> anyhow::Result<Uuid> {
                let vec_id = Uuid::now_v7();
                vectors.insert(
                    bitcode::encode(vec_id.as_bytes()),
                    bitcode::encode(embedding),
                )?;

                for kv in trees.iter() {
                    let (k, v) = kv?;
                    let id = Uuid::from_bytes(bitcode::decode::<[u8; 16]>(&k)?);
                    let mut tree: Node<N> = bitcode::decode(&v)?;
                    self.insert(&mut tree, embedding, vec_id)?;
                    trees.insert(bitcode::encode(id.as_bytes()), bitcode::encode(&tree))?;
                }

                Ok(vec_id)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        KEYSPACE.persist(PersistMode::SyncAll)?;
        Ok(vector_ids)
    }

    pub fn search_approximate<Met: Metric<Embedding<N>, Unit = DistanceUnit> + Send + Sync>(
        &self,
        query: Embedding<N>,
        top_k: usize,
        metric: &Met,
    ) -> anyhow::Result<Vec<(Uuid, DistanceUnit)>> {
        // using dashset instead of hashset as it support concurrent mutations to the set
        let candidates = DashSet::new();
        let trees = self.trees()?;

        for kv in trees.iter() {
            let (_, v) = kv?;
            let tree: Node<N> = bitcode::decode(&v)?;
            self.tree_result(query, top_k as i32, &tree, &candidates, metric)?;
        }

        let embeddings = self.embeddings()?;
        let mut sorted_candidates = candidates
            .into_par_iter()
            .map(|idx| {
                (
                    idx,
                    metric.distance(&self.vector(&embeddings, &idx), &query),
                )
            })
            .collect::<Vec<_>>();
        sorted_candidates.par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(sorted_candidates.into_iter().take(top_k as usize).collect())
    }
}
