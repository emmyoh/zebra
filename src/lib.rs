use fastembed::{Embedding, TextEmbedding};
use hnsw::Params;
use hnsw::{Hnsw, Searcher};
use pcg_rand::Pcg32;
use serde::Deserialize;
use serde::Serialize;
use simsimd::SpatialSimilarity;
use space::Metric;
use std::{error::Error, fs};

/// The path to the database on disk.
pub const DB_PATH: &str = "db";

/// A parameter regarding insertion into the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds.
pub const EF_CONSTRUCTION: usize = 400;

/// The candidate list size for the HNSW graph. Higher values result in more accurate search results at the expense of slower retrieval speeds.
pub const EF: usize = 24;

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
    pub hnsw: Hnsw<CosineSimilarity, Embedding, Pcg32, 12, 24>,
    /// The texts inserted into the database.
    pub texts: Vec<String>,
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
            let db = Database {
                hnsw,
                texts: Vec::new(),
            };
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

/// Insert texts into the database.
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
        db.hnsw.insert(embedding.clone(), &mut searcher);
        db.texts.push(text.clone());
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
    if db.texts.is_empty() {
        return Ok(Vec::new());
    }
    let mut searcher: Searcher<CosineDistance> = Searcher::default();
    let mut results = Vec::new();
    let model = TextEmbedding::try_new(Default::default())?;
    let query_embeddings = model.embed(texts, None)?;
    for query_embedding in query_embeddings.iter() {
        let mut neighbours = Vec::new();
        db.hnsw
            .nearest(&query_embedding, EF, &mut searcher, &mut neighbours);
        let nearest_neighbour = neighbours[0];
        results.push(db.texts[nearest_neighbour.index].clone());
    }
    Ok(results)
}
