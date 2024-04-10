use fastembed::{Embedding, TextEmbedding};
use hnsw::Params;
use hnsw::{Hnsw, Searcher};
use pcg_rand::Pcg64;
use serde::Deserialize;
use serde::Serialize;
use simsimd::SpatialSimilarity;
use space::{Metric, Neighbor};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter};
use std::{error::Error, fs};

/// The path to the database on disk.
pub const DB_PATH: &str = "db";

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds.
pub const EF_CONSTRUCTION: usize = 400;

/// The candidate list size for the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds.
pub const EF: usize = 24;

/// The number of bi-directional links created for each node in the HNSW graph.
pub const M: usize = 12;

/// The number of bi-directional links created for each node in the HNSW graph in the first layer.
pub const M0: usize = 24;

/// The data type representing the cosine similarity between two embeddings.
pub type CosineDistance = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// The cosine similarity distance metric.
pub struct CosineSimilarity;

impl Metric<Embedding> for CosineSimilarity {
    type Unit = CosineDistance;
    fn distance(&self, a: &Embedding, b: &Embedding) -> Self::Unit {
        let cosine_distance = 1.0 - f32::cosine(&a, &b).unwrap();
        cosine_distance.to_bits()
    }
}

#[derive(Serialize, Deserialize)]
/// A database containing texts and their embeddings.
pub struct Database {
    /// The Hierarchical Navigable Small World (HNSW) graph containing the embeddings.
    pub hnsw: Hnsw<CosineSimilarity, Embedding, Pcg64, M, M0>,
}

/// Save texts to disk.
///
/// # Arguments
///
/// * `texts` - A map of text indices and their corresponding texts.
pub fn save_texts_to_disk(texts: &mut HashMap<usize, String>) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all("texts")?;
    for text in texts {
        let mut reader = BufReader::new(text.1.as_bytes());
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!("texts/{}.lz4", text.0))?;
        let buf = BufWriter::new(file);
        let mut compressor = lz4_flex::frame::FrameEncoder::new(buf);
        io::copy(&mut reader, &mut compressor)?;
        compressor.finish()?;
    }
    Ok(())
}

/// Read texts from disk.
///
/// # Arguments
///
/// * `texts` - A vector of text indices to be read.
///
/// # Returns
///
/// A map of text indices and their corresponding texts.
pub fn read_texts_from_disk(
    texts: &mut Vec<usize>,
) -> Result<HashMap<usize, String>, Box<dyn Error>> {
    let mut results = HashMap::new();
    for text_index in texts {
        let file = OpenOptions::new()
            .read(true)
            .open(format!("texts/{}.lz4", text_index))?;
        let buf = BufReader::new(file);
        let mut decompressor = lz4_flex::frame::FrameDecoder::new(buf);
        let mut writer = BufWriter::new(Vec::new());
        io::copy(&mut decompressor, &mut writer)?;
        let text = String::from_utf8(writer.into_inner()?)?;
        results.insert(*text_index, text);
    }
    Ok(results)
}

/// Load the database from disk, or create it if it does not already exist.
///
/// # Returns
///
/// A database containing a HNSW graph and the inserted texts.
pub fn create_or_load_database() -> Result<Database, Box<dyn Error>> {
    let db_bytes = fs::read(DB_PATH);
    match db_bytes {
        Ok(bytes) => {
            let db: Database = bincode::deserialize(&bytes)?;
            Ok(db)
        }
        Err(_) => {
            let hnsw = Hnsw::new_params(
                CosineSimilarity,
                Params::new().ef_construction(EF_CONSTRUCTION),
            );
            let db = Database { hnsw };
            let db_bytes = bincode::serialize(&db)?;
            fs::write(DB_PATH, db_bytes)?;
            Ok(db)
        }
    }
}

/// Save the database to disk.
///
/// # Arguments
///
/// * `db` - The state of the database to be saved.
pub fn save_database(db: Database) -> Result<(), Box<dyn Error>> {
    let db_bytes = bincode::serialize(&db)?;
    fs::write(DB_PATH, db_bytes)?;
    Ok(())
}

/// Insert texts into the database. Inserting too many texts at once may take too much time and memory.
///
/// # Arguments
///
/// * `texts` - A vector of texts to be inserted.
///
/// # Returns
///
/// A tuple containing the number of embeddings inserted and the dimension of the embeddings.
pub fn insert_texts(texts: &mut Vec<String>) -> Result<(usize, usize), Box<dyn Error>> {
    texts.dedup();
    let model = TextEmbedding::try_new(Default::default())?;
    let new_embeddings: Vec<Embedding> = model.embed(texts.clone(), None)?;
    let length_and_dimension = (new_embeddings.len(), new_embeddings[0].len());
    let mut db = create_or_load_database()?;
    let mut searcher: Searcher<CosineDistance> = Searcher::default();
    for (text, embedding) in texts.iter().zip(new_embeddings.iter()) {
        let embedding_index = db.hnsw.insert(embedding.clone(), &mut searcher);
        let mut text_map = HashMap::new();
        text_map.insert(embedding_index, text.clone());
        save_texts_to_disk(&mut text_map)?;
    }
    save_database(db)?;
    Ok(length_and_dimension)
}

/// Query texts from the database.
///
/// # Arguments
///
/// * `texts` - A vector of texts to be queried.
///
/// # Returns
///
/// A vector of texts that are most similar to the queried texts.
pub fn query_texts<S: AsRef<str> + Send + Sync>(
    texts: Vec<S>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let db = create_or_load_database()?;
    if db.hnsw.is_empty() {
        return Ok(Vec::new());
    }
    let mut searcher: Searcher<CosineDistance> = Searcher::default();
    let mut results = Vec::new();
    let model = TextEmbedding::try_new(Default::default())?;
    let query_embeddings = model.embed(texts, None)?;
    for query_embedding in query_embeddings.iter() {
        let mut neighbours = [Neighbor {
            index: !0,
            distance: !0,
        }; EF];
        db.hnsw
            .nearest(&query_embedding, EF, &mut searcher, &mut neighbours);
        if neighbours.is_empty() {
            return Ok(Vec::new());
        }
        neighbours.sort_unstable_by_key(|n| n.distance);
        let nearest_neighbour = neighbours[0];
        results.push(nearest_neighbour.index);
    }
    let texts: Vec<String> = read_texts_from_disk(&mut results)?
        .values()
        .cloned()
        .collect();
    Ok(texts)
}
