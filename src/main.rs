use clap::{command, Parser, Subcommand};
use pretty_duration::pretty_duration;
use std::error::Error;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use text_db::{insert_texts, query_texts};
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let mut buffer = BufWriter::new(stdout().lock());
    let mut sw = Stopwatch::start_new();
    match cli.command {
        Some(Commands::Insert { mut texts }) => {
            writeln!(buffer, "Inserting {} text(s).", texts.len())?;
            let insertion_results = insert_texts(&mut texts)?;
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
            let num_texts = file_paths.len();
            writeln!(buffer, "Inserting texts from {} file(s).", num_texts)?;
            let mut i = 0;
            let mut texts = Vec::new();
            // Insert texts in batches of INSERT_BATCH_SIZE.
            for file_path in file_paths {
                let text = std::fs::read_to_string(file_path)?;
                texts.push(text);
                if i == INSERT_BATCH_SIZE - 1 {
                    let insertion_results = insert_texts(&mut texts)?;
                    writeln!(
                        buffer,
                        "{} embeddings of {} dimensions inserted into the database.",
                        insertion_results.0, insertion_results.1
                    )?;
                    texts.clear();
                    i = 0;
                } else {
                    i = i + 1;
                }
            }
            // Insert the remaining texts, if any.
            if !texts.is_empty() {
                let insertion_results = insert_texts(&mut texts)?;
                writeln!(
                    buffer,
                    "{} embeddings of {} dimensions inserted into the database.",
                    insertion_results.0, insertion_results.1
                )?;
            }
            sw.stop();
            writeln!(
                buffer,
                "Inserted {} text(s) in {}.",
                num_texts,
                pretty_duration(&sw.elapsed(), None)
            )?;
        }
        Some(Commands::Query { texts }) => {
            let num_texts = texts.len();
            writeln!(buffer, "Querying {} text(s).", num_texts)?;
            let query_results = query_texts(texts)?;
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
        _ => unreachable!(),
    }
    Ok(())
}
