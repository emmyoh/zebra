use clap::{command, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressDrawTarget};
use pretty_duration::pretty_duration;
use std::error::Error;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use ticky::Stopwatch;

const INSERT_BATCH_SIZE: usize = 100;

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help(true))]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    #[command(
        about = "Insert texts into the database.",
        arg_required_else_help(true)
    )]
    Insert { texts: Vec<String> },
    #[command(
        about = "Insert texts into the database from files on disk.",
        arg_required_else_help(true)
    )]
    InsertFromFiles { file_paths: Vec<PathBuf> },
    #[command(about = "Query texts from the database.", arg_required_else_help(true))]
    Query { texts: Vec<String> },
    #[command(about = "Clear the database.")]
    Clear,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let mut sw = Stopwatch::start_new();
    match cli.command {
        Some(Commands::Insert { mut texts }) => {
            let mut db = text_db::text::create_or_load_database()?;
            let mut buffer = BufWriter::new(stdout().lock());
            writeln!(buffer, "Inserting {} text(s).", texts.len())?;
            let insertion_results = db.insert_documents(&mut texts)?;
            sw.stop();
            writeln!(
                buffer,
                "{} embeddings of {} dimensions inserted into the database in {}.",
                insertion_results.0,
                insertion_results.1,
                pretty_duration(&sw.elapsed(), None)
            )?;
        }
        Some(Commands::InsertFromFiles { file_paths }) => {
            let mut db = text_db::text::create_or_load_database()?;
            let num_texts = file_paths.len();
            // writeln!(buffer, "Inserting texts from {} file(s).", num_texts)?;
            let progress_bar = ProgressBar::with_draw_target(
                Some(num_texts.try_into()?),
                ProgressDrawTarget::hidden(),
            )
            .with_message(format!("Inserting texts from {} file(s).", num_texts));
            let mut i = 0;
            let mut texts = Vec::new();
            // Insert texts in batches of INSERT_BATCH_SIZE.
            for file_path in file_paths {
                let text = std::fs::read_to_string(file_path)?;
                texts.push(text);
                if i == INSERT_BATCH_SIZE - 1 {
                    let insertion_results = db.insert_documents(&mut texts)?;
                    progress_bar.println(format!(
                        "{} embeddings of {} dimensions inserted into the database.",
                        insertion_results.0, insertion_results.1
                    ));
                    texts.clear();
                    i = 0;
                } else {
                    i += 1;
                }
                progress_bar.inc(1);
                if progress_bar.is_hidden() {
                    progress_bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(100));
                }
            }
            // Insert the remaining texts, if any.
            if !texts.is_empty() {
                let insertion_results = db.insert_documents(&mut texts)?;
                progress_bar.println(format!(
                    "{} embeddings of {} dimensions inserted into the database.",
                    insertion_results.0, insertion_results.1
                ));
            }
            sw.stop();
            progress_bar.println(format!(
                "Inserted {} text(s) in {}.",
                num_texts,
                pretty_duration(&sw.elapsed(), None)
            ));
        }
        Some(Commands::Query { texts }) => {
            let mut db = text_db::text::create_or_load_database()?;
            let mut buffer = BufWriter::new(stdout().lock());
            let num_texts = texts.len();
            writeln!(buffer, "Querying {} text(s).", num_texts)?;
            let query_results = db.query_documents(texts)?;
            sw.stop();
            writeln!(
                buffer,
                "Queried {} text(s) in {}.",
                num_texts,
                pretty_duration(&sw.elapsed(), None)
            )?;
            writeln!(buffer, "Results:")?;
            for result in query_results {
                writeln!(buffer, "1. \t{}", result)?;
            }
        }
        Some(Commands::Clear) => {
            let mut buffer = BufWriter::new(stdout().lock());
            writeln!(buffer, "Clearing database.")?;
            std::fs::remove_file("text.db").unwrap_or(());
            std::fs::remove_dir_all("texts").unwrap_or(());
            std::fs::remove_dir_all(".fastembed_cache").unwrap_or(());
            sw.stop();
            writeln!(
                buffer,
                "Database cleared in {}.",
                pretty_duration(&sw.elapsed(), None)
            )?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
