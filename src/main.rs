use clap::{command, Parser, Subcommand};
use std::error::Error;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use text_db::{insert_texts, query_texts};

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
    match cli.command {
        Some(Commands::Insert { mut texts }) => {
            writeln!(buffer, "Inserting {} text(s).", texts.len())?;
            let insertion_results = insert_texts(&mut texts)?;
            writeln!(
                buffer,
                "{} embeddings of {} dimensions inserted into the database.",
                insertion_results.0, insertion_results.1
            )?;
        }
        Some(Commands::InsertFromFiles { file_paths }) => {
            writeln!(buffer, "Inserting texts from {} file(s).", file_paths.len())?;
            let mut texts = Vec::new();
            for file_path in file_paths {
                let text = std::fs::read_to_string(file_path)?;
                texts.push(text);
            }
            let insertion_results = insert_texts(&mut texts)?;
            writeln!(
                buffer,
                "{} embeddings of {} dimensions inserted into the database.",
                insertion_results.0, insertion_results.1
            )?;
        }
        Some(Commands::Query { texts }) => {
            writeln!(buffer, "Querying {} text(s).", texts.len())?;
            let query_results = query_texts(texts)?;
            writeln!(buffer, "Results: {:?}", query_results)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
