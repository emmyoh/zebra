use crate::index::lsh::ANNIndex;
use crate::Embedding;
use crate::{distance::DistanceUnit, model::core::DatabaseEmbeddingModel};
use bitcode::{Decode, Encode};
use bytes::Bytes;
use dashmap::{DashMap, DashSet};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use space::Metric;
use std::io::{Cursor, Read};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufReader, BufWriter},
};
use uuid::Uuid;

#[derive(Encode, Decode)]
/// A database containing documents and their embeddings.
///
/// # Arguments
///
/// * `Met` - The distance metric for the embeddings. Can be changed after database creation.
///
/// * `Mod` - The model used to generate embeddings. Should not be changed after database creation.
pub struct Database<
    const N: usize,
    Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Encode + Send + Sync,
    Mod: DatabaseEmbeddingModel<N> + Default + Encode + Send + Sync,
> {
    uuid: [u8; 16],
    /// The model used to generate embeddings. Should not be changed after database creation.
    pub model: Mod,
    pub metric: Met,
    index: ANNIndex<N>,
    path: String,
}

impl<
        const N: usize,
        Met: Metric<Embedding<N>, Unit = DistanceUnit> + Default + Encode + Send + Sync,
        Mod: DatabaseEmbeddingModel<N> + Default + Encode + Send + Sync,
    > Database<N, Met, Mod>
where
    for<'de> Mod: Decode<'de>,
    for<'de> Met: Decode<'de>,
{
    fn database_subdirectory(&self) -> String {
        format!("{}", Uuid::from_bytes(self.uuid).as_simple())
    }

    fn default_database_path(&self) -> String {
        format!("{}.zebra", Uuid::from_bytes(self.uuid).as_simple())
    }

    /// Load the database from disk.
    ///
    /// # Returns
    ///
    /// A database containing embeddings & documents.
    pub fn open(path: &String) -> anyhow::Result<Self> {
        let db_bytes = fs::read(path)?;
        let mut db: Self = bitcode::decode(&db_bytes)?;
        db.path = path.clone();
        Ok(db)
    }

    /// Create a database in memory.
    ///
    /// # Returns
    ///
    /// A Zebra database.
    pub fn new() -> Self {
        Self {
            uuid: Uuid::now_v7().into_bytes(),
            model: Mod::default(),
            metric: Met::default(),
            index: ANNIndex::build_index(15),
            path: String::new(),
        }
    }

    /// Load the database from disk, or create it if it does not already exist.
    ///
    /// # Returns
    ///
    /// A database containing embeddings & documents.
    pub fn open_or_create(path: &String) -> Self {
        Self::open(path).unwrap_or(Self::new())
    }

    /// Save the database to disk.
    pub fn save_database(&self, path: Option<&String>) -> anyhow::Result<()> {
        fs::write(path.unwrap_or(&self.path), bitcode::encode(self))?;
        Ok(())
    }

    /// Delete the database.
    pub fn clear_database(&self) {
        let _ = std::fs::remove_file(self.default_database_path());
        let _ = std::fs::remove_dir_all(self.database_subdirectory());
    }

    /// Insert documents into the database. Inserting too many documents at once may take too much time and memory.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be inserted.
    ///
    /// # Returns
    ///
    /// A tuple containing the number of embeddings inserted and the dimension of the embeddings.
    pub fn insert_documents(&self, documents: &Vec<Bytes>) -> anyhow::Result<(usize, usize)> {
        let new_embeddings: Vec<Embedding<N>> = self.model.embed_documents(documents)?;
        let length_and_dimension = (new_embeddings.len(), new_embeddings[0].len());
        self.insert_records(&new_embeddings, documents)?;
        Ok(length_and_dimension)
    }

    /// Insert embedding-byte pairs into the database.
    ///
    /// # Arguments
    ///
    /// * `records` - A list of embeddings & the raw bytes they point to, presumably being the embedded documents.
    pub fn insert_records(
        &self,
        embeddings: &Vec<Embedding<N>>,
        documents: &Vec<Bytes>,
    ) -> anyhow::Result<()> {
        let embedding_ids = self.index.add(embeddings)?;
        self.save_documents_to_disk(&embedding_ids, &documents)?;
        self.save_database(None)?;
        Ok(())
    }

    /// Query documents from the database.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be queried.
    ///
    /// * `number_of_results` - The candidate list size for the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds.
    ///
    /// # Returns
    ///
    /// A vector of documents that are most similar to the queried documents.
    pub fn query_documents(
        &self,
        documents: &Vec<Bytes>,
        number_of_results: usize,
    ) -> anyhow::Result<DashMap<Uuid, Vec<u8>>> {
        if self.index.is_empty() {
            return Ok(DashMap::new());
        }
        let results = DashSet::new();
        let query_embeddings = self.model.embed_documents(documents)?;
        query_embeddings.into_par_iter().for_each(|x| {
            let mut neighbours = self
                .index
                .search_approximate(x, number_of_results, &self.metric)
                .unwrap_or_default();
            neighbours.par_sort_unstable_by_key(|n| n.1);
            for neighbour in neighbours {
                results.insert(neighbour.0);
            }
        });
        Ok(self.read_documents_from_disk(&results)?)
    }

    /// Save documents to disk.
    ///
    /// # Arguments
    ///
    /// * `documents` - A map of document indices and their corresponding documents.
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
    /// * `documents` - A vector of document indices to be read.
    ///
    /// # Returns
    ///
    /// A map of document indices and the bytes of their corresponding documents.
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
