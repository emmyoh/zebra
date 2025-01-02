/// Core implementation of a database.
pub mod core;
#[cfg(feature = "default_db")]
/// Default configurations of databases.
pub mod default;
/// Implementations of database indices.
pub mod index;
