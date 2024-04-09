use fastembed::{Embedding, TextEmbedding};
use simsimd::SpatialSimilarity;
use std::{collections::HashMap, error::Error, fs};

/// The path to the database on disk.
pub const DB_PATH: &str = "db";

/// Load texts & their embeddings from the database.
///
/// # Returns
///
/// A dictionary containing embeddings accessible by their text.
pub fn load_text_embeddings() -> Result<HashMap<String, Embedding>, Box<dyn Error>> {
    let db_bytes = fs::read(DB_PATH)?;
    let texts_by_embeddings: HashMap<String, Embedding> = bincode::deserialize(&db_bytes)?;
    Ok(texts_by_embeddings)
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
pub fn insert_texts(texts: Vec<String>) -> Result<(usize, usize), Box<dyn Error>> {
    let model = TextEmbedding::try_new(Default::default())?;
    let new_embeddings = model.embed(texts.clone(), None)?;
    let length_and_dimension = (new_embeddings.len(), new_embeddings[0].len());
    let new_text_embeddings = texts
        .iter()
        .zip(new_embeddings.iter())
        .map(|(text, embedding)| (text.to_string(), embedding.clone()));
    let mut text_embeddings = load_text_embeddings()?;
    text_embeddings.extend(new_text_embeddings);
    let db_bytes = bincode::serialize(&text_embeddings)?;
    fs::write(DB_PATH, db_bytes)?;
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
/// A vector of embeddings.
pub fn query_texts<S: AsRef<str> + Send + Sync>(
    texts: Vec<S>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let text_embeddings = load_text_embeddings()?;
    let model = TextEmbedding::try_new(Default::default())?;
    let query_embeddings = model.embed(texts, None)?;
    let mut results = Vec::new();
    for query_embedding in query_embeddings {
        let similarity_scores = text_embeddings
            .iter()
            .map(|text_embedding| f32::cosine(&query_embedding, &text_embedding.1).unwrap())
            .collect::<Vec<f64>>();
        let max_similarity = similarity_scores
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        let max_similarity_index = similarity_scores
            .iter()
            .position(|r| r == max_similarity)
            .unwrap();
        results.push(
            text_embeddings
                .keys()
                .nth(max_similarity_index)
                .unwrap()
                .to_string(),
        );
    }
    Ok(results)
}
