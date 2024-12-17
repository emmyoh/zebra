use crate::Embedding;
use crate::{distance::DistanceUnit, model::core::DatabaseEmbeddingModel};
use bytes::Bytes;
use hnsw::{Hnsw, Params, Searcher};
use pcg_rand::Pcg64;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use space::Metric;
use std::{
    collections::HashMap,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufReader, BufWriter},
};

/// Retrieve the path to the database file.
///
/// # Returns
///
/// The path to the database file.
fn database_path(
    ef_construction: usize,
    m: usize,
    m0: usize,
    model: &impl DatabaseEmbeddingModel,
) -> String {
    format!(
        "{}.zebra",
        database_subdirectory(ef_construction, m, m0, model)
    )
}

/// Retrieve the path to the database document directory.
///
/// # Returns
///
/// The path to the database document directory.
fn database_subdirectory(
    ef_construction: usize,
    m: usize,
    m0: usize,
    model: &impl DatabaseEmbeddingModel,
) -> String {
    filenamify::filenamify(format!(
        "{}_{}_{}_{}_{}",
        ef_construction,
        m,
        m0,
        model.document_type(),
        model.typetag_name()
    ))
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
/// The type of document stored in the database.
pub enum DocumentType {
    /// A document containing text.
    Text,
    /// A document containing an image.
    Image,
    /// A document containing audio.
    Audio,
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DocumentType::Text => "Text",
                DocumentType::Image => "Image",
                DocumentType::Audio => "Audio",
            }
        )
    }
}

#[derive(Serialize, Deserialize)]
/// A database containing documents and their embeddings.
///
/// # Arguments
///
/// * `Met` - The distance metric for the embeddings. Can be changed after database creation.
///
/// * `Mod` - The model used to generate embeddings. Should not be changed after database creation.
///
/// * `EF_CONSTRUCTION` - A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
///
/// * `M` - The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
///
/// * `M0` - The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub struct Database<
    Met: Metric<Embedding, Unit = DistanceUnit> + Default + Serialize,
    Mod: DatabaseEmbeddingModel + Default + Serialize,
    const EF_CONSTRUCTION: usize,
    const M: usize,
    const M0: usize,
> {
    /// The Hierarchical Navigable Small World (HNSW) graph containing the embeddings.
    pub hnsw: Hnsw<Met, Embedding, Pcg64, M, M0>,
    /// The model used to generate embeddings. Should not be changed after database creation.
    pub model: Mod,
}

impl<
        Met: Metric<Embedding, Unit = DistanceUnit> + Default + Serialize,
        Mod: DatabaseEmbeddingModel + Default + Serialize,
        const EF_CONSTRUCTION: usize,
        const M: usize,
        const M0: usize,
    > Database<Met, Mod, EF_CONSTRUCTION, M, M0>
where
    for<'de> Met: Deserialize<'de>,
    for<'de> Mod: Deserialize<'de>,
{
    fn database_subdirectory(&self) -> String {
        database_subdirectory(EF_CONSTRUCTION, M, M0, &Mod::default())
    }

    fn database_path(&self) -> String {
        database_path(EF_CONSTRUCTION, M, M0, &Mod::default())
    }

    /// Load the database from disk.
    ///
    /// # Returns
    ///
    /// A database containing a HNSW graph and inserted documents.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let db_path = database_path(EF_CONSTRUCTION, M, M0, &Mod::default());
        let db_bytes = fs::read(db_path.clone())?;
        Ok(bincode::deserialize(&db_bytes)?)
    }

    /// Create a database in memory.
    ///
    /// # Returns
    ///
    /// A Zebra database.
    pub fn new() -> Self {
        let hnsw = Hnsw::new_params(
            Met::default(),
            Params::new().ef_construction(EF_CONSTRUCTION),
        );
        Self {
            hnsw,
            model: Mod::default(),
        }
    }

    /// Load the database from disk, or create it if it does not already exist.
    ///
    /// # Returns
    ///
    /// A database containing a HNSW graph and inserted documents.
    pub fn create_or_load_database() -> Self {
        Self::load().unwrap_or(Self::new())
    }

    /// Save the database to disk.
    pub fn save_database(&self) -> Result<(), Box<dyn Error>> {
        let db_bytes = bincode::serialize(&self)?;
        fs::write(self.database_path(), db_bytes)?;
        Ok(())
    }

    /// Delete the database.
    pub fn clear_database(&self) {
        let _ = std::fs::remove_file(self.database_path());
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
    pub fn insert_documents(
        &mut self,
        documents: Vec<Bytes>,
    ) -> Result<(usize, usize), Box<dyn Error>> {
        let new_embeddings: Vec<Embedding> = self.model.embed_documents(documents.to_vec())?;
        let length_and_dimension = (new_embeddings.len(), new_embeddings[0].len());
        let records: Vec<_> = new_embeddings
            .into_par_iter()
            .zip(documents.into_par_iter())
            .collect();
        self.insert_records(records)?;
        Ok(length_and_dimension)
    }

    /// Insert embedding-byte pairs into the database.
    ///
    /// # Arguments
    ///
    /// * `records` - A list of embeddings & the raw bytes they point to, presumably being the embedded documents.
    pub fn insert_records(
        &mut self,
        records: Vec<(Embedding, Bytes)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut searcher: Searcher<DistanceUnit> = Searcher::default();
        let mut document_map = HashMap::new();
        for (embedding, document) in records {
            let embedding_index = self.hnsw.insert(embedding.clone(), &mut searcher);
            document_map.insert(embedding_index, document.clone());
        }
        self.save_documents_to_disk(&mut document_map)?;
        self.save_database()?;
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
        &mut self,
        documents: Vec<Bytes>,
        number_of_results: usize,
    ) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        if self.hnsw.is_empty() {
            return Ok(Vec::new());
        }
        let mut searcher: Searcher<DistanceUnit> = Searcher::default();
        let mut results = Vec::new();
        let query_embeddings = self.model.embed_documents(documents)?;
        for query_embedding in query_embeddings.iter() {
            let mut neighbours = Vec::new();
            self.hnsw.nearest(
                query_embedding,
                number_of_results,
                &mut searcher,
                &mut neighbours,
            );
            if neighbours.is_empty() {
                return Ok(Vec::new());
            }
            neighbours.sort_unstable_by_key(|n| n.distance);
            for i in 0..number_of_results {
                let neighbour = neighbours[i];
                results.push(neighbour.index);
            }
        }
        let documents = self
            .read_documents_from_disk(&mut results)?
            .values()
            .cloned()
            .collect();
        Ok(documents)
    }

    /// Save documents to disk.
    ///
    /// # Arguments
    ///
    /// * `documents` - A map of document indices and their corresponding documents.
    pub fn save_documents_to_disk(
        &self,
        documents: &mut HashMap<usize, Bytes>,
    ) -> Result<(), Box<dyn Error>> {
        let document_subdirectory = self.database_subdirectory();
        std::fs::create_dir_all(document_subdirectory.clone())?;
        for document in documents {
            let mut reader = BufReader::new(document.1.as_ref());
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(format!("{}/{}.lz4", document_subdirectory, document.0))?;
            let buf = BufWriter::new(file);
            let mut compressor = lz4_flex::frame::FrameEncoder::new(buf);
            io::copy(&mut reader, &mut compressor)?;
            compressor.finish()?;
        }
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
        documents: &mut Vec<usize>,
    ) -> Result<HashMap<usize, Vec<u8>>, Box<dyn Error>> {
        let document_subdirectory = self.database_subdirectory();
        let mut results = HashMap::new();
        for document_index in documents {
            let file = OpenOptions::new()
                .read(true)
                .open(format!("{}/{}.lz4", document_subdirectory, document_index))?;
            let buf = BufReader::new(file);
            let mut decompressor = lz4_flex::frame::FrameDecoder::new(buf);
            let mut writer = BufWriter::new(Vec::new());
            io::copy(&mut decompressor, &mut writer)?;
            let document = writer.into_inner()?;
            results.insert(*document_index, document);
        }
        Ok(results)
    }
}
