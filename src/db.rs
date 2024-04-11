use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use hnsw::Params;
use hnsw::{Hnsw, Searcher};
use pcg_rand::Pcg64;
use serde::{Deserialize, Serialize};
use space::{Metric, Neighbor};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter};
use std::usize;
use std::{error::Error, fs};

use crate::distance::DistanceUnit;
use crate::EF;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
/// The type of document stored in the database.
pub enum DocumentType {
    /// A document containing text.
    Text,
}

impl DocumentType {
    /// Get the name of the subdirectory containing the documents of this type.
    ///
    /// # Returns
    ///
    /// The name of the subdirectory.
    pub fn subdirectory_name(&self) -> &str {
        match self {
            DocumentType::Text => "texts",
        }
    }

    /// Get the name of the database file containing the documents of this type.
    ///
    /// # Returns
    ///
    /// The name of the database file.
    pub fn database_name(&self) -> &str {
        match self {
            DocumentType::Text => "text.db",
        }
    }
}

#[derive(Serialize, Deserialize)]
/// A database containing documents and their embeddings.
///
/// # Arguments
///
/// * `Met` - The distance metric for the embeddings. Can be changed after database creation.
///
/// * `EF_CONSTRUCTION` - A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds. Cannot be changed after database creation.
///
/// * `M` - The number of bi-directional links created for each node in the HNSW graph. Cannot be changed after database creation. Increases memory usage and decreases retrieval speed with higher values.
///
/// * `M0` - The number of bi-directional links created for each node in the HNSW graph in the first layer. Cannot be changed after database creation.
pub struct Database<
    Met: Metric<Embedding> + Serialize,
    const EF_CONSTRUCTION: usize,
    const M: usize,
    const M0: usize,
> {
    /// The Hierarchical Navigable Small World (HNSW) graph containing the embeddings.
    pub hnsw: Hnsw<Met, Embedding, Pcg64, M, M0>,
    /// The type of documents stored in the database.
    pub document_type: DocumentType,
}

impl<
        Met: Metric<Embedding, Unit = DistanceUnit> + Serialize,
        const EF_CONSTRUCTION: usize,
        const M: usize,
        const M0: usize,
    > Database<Met, EF_CONSTRUCTION, M, M0>
where
    for<'de> Met: Deserialize<'de>,
{
    /// Load the database from disk, or create it if it does not already exist.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance metric for the embeddings.
    ///
    /// # Returns
    ///
    /// A database containing a HNSW graph and the inserted documents.
    pub fn create_or_load_database(
        metric: Met,
        document_type: DocumentType,
    ) -> Result<Self, Box<dyn Error>> {
        let db_bytes = fs::read(document_type.database_name());
        match db_bytes {
            Ok(bytes) => {
                let db: Self = bincode::deserialize(&bytes)?;
                Ok(db)
            }
            Err(_) => {
                let hnsw = Hnsw::new_params(metric, Params::new().ef_construction(EF_CONSTRUCTION));
                let db = Database {
                    hnsw,
                    document_type,
                };
                let db_bytes = bincode::serialize(&db)?;
                fs::write(document_type.database_name(), db_bytes)?;
                Ok(db)
            }
        }
    }

    /// Save the database to disk.
    pub fn save_database(&self) -> Result<(), Box<dyn Error>> {
        let db_bytes = bincode::serialize(&self)?;
        fs::write(self.document_type.database_name(), db_bytes)?;
        Ok(())
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
        documents: &mut Vec<String>,
    ) -> Result<(usize, usize), Box<dyn Error>> {
        documents.dedup();
        let model = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::BGESmallENV15,
            show_download_progress: false,
            ..Default::default()
        })?;
        let new_embeddings: Vec<Embedding> = model.embed(documents.clone(), None)?;
        let length_and_dimension = (new_embeddings.len(), new_embeddings[0].len());
        let mut searcher: Searcher<DistanceUnit> = Searcher::default();
        for (text, embedding) in documents.iter().zip(new_embeddings.iter()) {
            let embedding_index = self.hnsw.insert(embedding.clone(), &mut searcher);
            let mut text_map = HashMap::new();
            text_map.insert(embedding_index, text.clone());
            self.save_documents_to_disk(&mut text_map)?;
        }
        self.save_database()?;
        Ok(length_and_dimension)
    }

    /// Query documents from the database.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be queried.
    ///
    /// # Returns
    ///
    /// A vector of documents that are most similar to the queried documents.
    pub fn query_documents<S: AsRef<str> + Send + Sync>(
        &mut self,
        documents: Vec<S>,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if self.hnsw.is_empty() {
            return Ok(Vec::new());
        }
        let mut searcher: Searcher<DistanceUnit> = Searcher::default();
        let mut results = Vec::new();
        let model = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::BGESmallENV15,
            show_download_progress: false,
            ..Default::default()
        })?;
        let query_embeddings = model.embed(documents, None)?;
        for query_embedding in query_embeddings.iter() {
            let mut neighbours = [Neighbor {
                index: !0,
                distance: !0,
            }; EF];
            self.hnsw
                .nearest(query_embedding, EF, &mut searcher, &mut neighbours);
            if neighbours.is_empty() {
                return Ok(Vec::new());
            }
            neighbours.sort_unstable_by_key(|n| n.distance);
            let nearest_neighbour = neighbours[0];
            results.push(nearest_neighbour.index);
        }
        let documents: Vec<String> = self
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
    /// * `documents` - A map of text indices and their corresponding documents.
    pub fn save_documents_to_disk(
        &self,
        documents: &mut HashMap<usize, String>,
    ) -> Result<(), Box<dyn Error>> {
        let document_subdirectory = self.document_type.subdirectory_name();
        std::fs::create_dir_all(document_subdirectory)?;
        for text in documents {
            let mut reader = BufReader::new(text.1.as_bytes());
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(format!("{}/{}.lz4", document_subdirectory, text.0))?;
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
    /// * `documents` - A vector of text indices to be read.
    ///
    /// # Returns
    ///
    /// A map of text indices and their corresponding documents.
    pub fn read_documents_from_disk(
        &self,
        documents: &mut Vec<usize>,
    ) -> Result<HashMap<usize, String>, Box<dyn Error>> {
        let document_subdirectory = self.document_type.subdirectory_name();
        let mut results = HashMap::new();
        for text_index in documents {
            let file = OpenOptions::new()
                .read(true)
                .open(format!("{}/{}.lz4", document_subdirectory, text_index))?;
            let buf = BufReader::new(file);
            let mut decompressor = lz4_flex::frame::FrameDecoder::new(buf);
            let mut writer = BufWriter::new(Vec::new());
            io::copy(&mut decompressor, &mut writer)?;
            let text = String::from_utf8(writer.into_inner()?)?;
            results.insert(*text_index, text);
        }
        Ok(results)
    }
}
