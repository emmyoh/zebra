use super::index::lsh::{LSHIndex, LSHIndexOptions};
use crate::Embedding;
use crate::{distance::DistanceUnit, model::core::DatabaseEmbeddingModel};
use bytes::Bytes;
use dashmap::{DashMap, DashSet};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use space::Metric;
use std::io::Cursor;
use std::{
    fs::{self, OpenOptions},
    io::{self, BufReader, BufWriter},
};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatabaseInner<
    const N: usize,
    Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Serialize + Send + Sync,
    Mod: DatabaseEmbeddingModel<N> + Default + Serialize + Send + Sync,
> {
    uuid: Uuid,
    model: Mod,
    metric: Met,
    index_options: LSHIndexOptions<N>,
}

impl<
        const N: usize,
        Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Serialize + Send + Sync,
        Mod: DatabaseEmbeddingModel<N> + Default + Serialize + Send + Sync,
    > DatabaseInner<N, Met, Mod>
where
    for<'de> Mod: Deserialize<'de>,
    for<'de> Met: Deserialize<'de>,
{
    fn index(&self) -> anyhow::Result<LSHIndex<N>> {
        LSHIndex::new(&self.uuid, &self.index_options)
    }
}

#[derive(Clone)]
/// A database containing embedding vectors and documents.
///
/// # Arguments
///
/// * `N` - The dimensionality of the vectors in the database.
///
/// * `Met` - The distance metric used by the database index.
///
/// * `Mod` - The model used to generate the embedding vectors.
pub struct Database<
    const N: usize,
    Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Serialize + Send + Sync,
    Mod: DatabaseEmbeddingModel<N> + Default + Serialize + Send + Sync,
> {
    inner: DatabaseInner<N, Met, Mod>,
    /// The database index used to approximate nearest-neighbour search.
    pub index: LSHIndex<N>,
    path: String,
}

impl<
        const N: usize,
        Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Serialize + Send + Sync,
        Mod: DatabaseEmbeddingModel<N> + Default + Serialize + Send + Sync,
    > Database<N, Met, Mod>
where
    for<'de> Mod: Deserialize<'de>,
    for<'de> Met: Deserialize<'de>,
{
    fn database_subdirectory(&self) -> String {
        format!("{}", self.inner.uuid.as_simple())
    }

    fn default_database_path(&self) -> String {
        format!("{}.zebra", self.inner.uuid.as_simple())
    }

    /// Load the database from disk.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file.
    ///
    /// # Returns
    ///
    /// A [Database] containing embeddings & documents.
    pub fn open(path: &String) -> anyhow::Result<Self> {
        let db_bytes = fs::read(path)?;
        let inner: DatabaseInner<N, Met, Mod> =
            bincode::serde::decode_from_slice(&db_bytes, bincode::config::legacy())?.0;
        let index = inner.index()?;
        Ok(Self {
            inner,
            path: path.clone(),
            index,
        })
    }

    /// Create a database in memory.\
    /// Note: All database operations require read & write access to storage; a database cannot be used only in memory.
    ///
    /// # Returns
    ///
    /// A new [Database].
    pub fn new(index_options: &LSHIndexOptions<N>) -> anyhow::Result<Self> {
        let uuid = Uuid::now_v7();
        let inner = DatabaseInner {
            uuid,
            model: Mod::default(),
            metric: Met::default(),
            index_options: index_options.clone(),
        };
        let index = inner.index()?;
        let mut new = Self {
            inner,
            index,
            path: String::new(),
        };
        new.path = new.default_database_path();
        Ok(new)
    }

    /// Create a database in memory, persisting to storage at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file.
    ///
    /// # Returns
    ///
    /// A [Database] containing embeddings & documents.
    pub fn new_with_path(
        path: &String,
        index_options: &LSHIndexOptions<N>,
    ) -> anyhow::Result<Self> {
        let uuid = Uuid::now_v7();
        let inner = DatabaseInner {
            uuid,
            model: Mod::default(),
            metric: Met::default(),
            index_options: index_options.clone(),
        };
        let index = inner.index()?;
        Ok(Self {
            inner,
            index,
            path: path.to_owned(),
        })
    }

    /// Load the database from disk, or create it if it does not already exist.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file.
    ///
    /// # Returns
    ///
    /// A [Database] containing embeddings & documents.
    pub fn open_or_create(
        path: &String,
        index_options: &LSHIndexOptions<N>,
    ) -> anyhow::Result<Self> {
        Ok(Self::open(path).unwrap_or(Self::new_with_path(path, index_options)?))
    }

    /// Save the database to disk.
    ///
    /// # Arguments
    ///
    /// * `path` - An optional path to save the database to; if left blank, will use the path the database was opened from.
    pub fn save_database(&self, path: Option<&String>) -> anyhow::Result<()> {
        fs::write(
            path.unwrap_or(&self.path),
            bincode::serde::encode_to_vec(&self.inner, bincode::config::legacy())?,
        )?;
        Ok(())
    }

    /// Delete the database and its contents, including all vectors and documents.\
    /// Note: This deletes the file at the path the database was opened from; if the database file was moved after opening, this may have unintended consequences.
    pub fn clear_database(&self) {
        let _ = self.index.clear();
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir_all(self.database_subdirectory());
    }

    /// Removes records from the database.
    ///
    /// # Arguments
    ///
    /// * `embedding_ids` - The IDs of the vectors to remove.
    pub fn remove(&self, embedding_ids: &Vec<Uuid>) -> anyhow::Result<()> {
        let document_subdirectory = self.database_subdirectory();
        let removed = self.index.remove(embedding_ids)?;
        removed.into_par_iter().for_each(|x| {
            let _ = std::fs::remove_file(format!("{}/{}.lz4", document_subdirectory, x));
        });
        Ok(())
    }

    /// Remove duplicate embedding vectors from the database.
    pub fn deduplicate(&self) -> anyhow::Result<()> {
        let document_subdirectory = self.database_subdirectory();
        let removed = self.index.deduplicate()?;
        removed.into_par_iter().for_each(|x| {
            let _ = std::fs::remove_file(format!("{}/{}.lz4", document_subdirectory, x));
        });
        Ok(())
    }

    /// Insert documents into the database.\
    /// Consider batching insertions as inserting too many documents at once may be memory-intensive.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be inserted.
    pub fn insert_documents(&self, documents: &Vec<Bytes>) -> anyhow::Result<()> {
        let new_embeddings: Vec<Embedding<N>> = self.inner.model.embed_documents(documents)?;
        self.insert_records(&new_embeddings, documents)
    }

    /// Insert embedding-byte pairs into the database.\
    /// Consider batching insertions as inserting too many records at once may be memory-intensive.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - A list of embedding vectors to insert.
    ///
    /// * `documents` - A list of documents to pair with the embedding vectors.
    pub fn insert_records(
        &self,
        embeddings: &Vec<Embedding<N>>,
        documents: &Vec<Bytes>,
    ) -> anyhow::Result<()> {
        let embedding_ids = self.index.add(embeddings)?;
        self.save_documents_to_disk(&embedding_ids, documents)?;
        self.save_database(None)?;
        Ok(())
    }

    /// Query records from the database.
    ///
    /// # Arguments
    ///
    /// * `documents` - A list of query documents.
    ///
    /// * `number_of_results` - The maximum number of approximate nearest neighbours to return for each query document.
    ///
    /// # Returns
    ///
    /// The records for documents that are most similar to the query documents.
    pub fn query_documents(
        &self,
        documents: &[Bytes],
        number_of_results: usize,
    ) -> anyhow::Result<DashMap<usize, DashMap<Uuid, Vec<u8>>>> {
        if self.index.no_vectors() {
            return Ok(DashMap::new());
        }
        let query_embeddings = self.inner.model.embed_documents(documents)?;
        self.query_vectors(&query_embeddings, number_of_results)
    }

    /// Query records from the database.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A list of query vectors.
    ///
    /// * `number_of_results` - The maximum number of approximate nearest neighbours to return for each query vector.
    ///
    /// # Returns
    ///
    /// The records for documents that are most similar to the query vectors.
    pub fn query_vectors(
        &self,
        vectors: &Vec<Embedding<N>>,
        number_of_results: usize,
    ) -> anyhow::Result<DashMap<usize, DashMap<Uuid, Vec<u8>>>> {
        if self.index.no_vectors() {
            return Ok(DashMap::new());
        }
        let results = DashMap::new();
        vectors.into_par_iter().enumerate().for_each(|(idx, x)| {
            let mut neighbours = self
                .index
                .search(x, number_of_results, &self.inner.metric)
                .unwrap_or_default();
            neighbours.par_sort_unstable_by_key(|n| n.1);
            let neighbour_ids: DashSet<_> = neighbours.into_iter().map(|(id, _)| id).collect();
            results.insert(
                idx,
                self.read_documents_from_disk(&neighbour_ids)
                    .unwrap_or_default(),
            );
        });
        Ok(results)
    }

    /// Save documents to disk.
    ///
    /// # Arguments
    ///
    /// * `embedding_ids` - A list of document IDs to be inserted.
    ///
    /// * `documents` - A list of documents to be inserted.
    pub fn save_documents_to_disk(
        &self,
        embedding_ids: &Vec<Uuid>,
        documents: &Vec<Bytes>,
    ) -> anyhow::Result<()> {
        let document_subdirectory = self.database_subdirectory();
        std::fs::create_dir_all(document_subdirectory.clone())?;
        embedding_ids
            .par_iter()
            .zip(documents.par_iter())
            .map(|(id, document)| -> anyhow::Result<()> {
                let mut reader = BufReader::new(Cursor::new(document));
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(format!("{}/{}.lz4", document_subdirectory, id))?;
                let buf = BufWriter::new(file);
                let mut compressor = lz4_flex::frame::FrameEncoder::new(buf);
                io::copy(&mut reader, &mut compressor)?;
                compressor.finish()?;
                Ok(())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(())
    }

    /// Read documents from disk.
    ///
    /// # Arguments
    ///
    /// * `documents` - A set of document IDs to be read.
    ///
    /// # Returns
    ///
    /// A map of document IDs to the bytes of their corresponding documents.
    pub fn read_documents_from_disk(
        &self,
        documents: &DashSet<Uuid>,
    ) -> anyhow::Result<DashMap<Uuid, Vec<u8>>> {
        let document_subdirectory = self.database_subdirectory();
        let results = DashMap::new();
        documents
            .into_par_iter()
            .map(|document_index| -> anyhow::Result<()> {
                let file = OpenOptions::new().read(true).open(format!(
                    "{}/{}.lz4",
                    document_subdirectory,
                    document_index.as_simple()
                ))?;
                let buf = BufReader::new(file);
                let mut decompressor = lz4_flex::frame::FrameDecoder::new(buf);
                let mut writer = BufWriter::new(Vec::new());
                io::copy(&mut decompressor, &mut writer)?;
                let document = writer.into_inner()?;
                results.insert(*document_index, document);
                Ok(())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(results)
    }
}
