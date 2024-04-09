use clap::{command, Parser, Subcommand};
use std::error::Error;
use std::io::Write;
use std::io::{stdout, BufWriter};
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
    #[command(about = "Query texts from the database.", arg_required_else_help(true))]
    Query { texts: Vec<String> },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let mut buffer = BufWriter::new(stdout().lock());
    match cli.command {
        Some(Commands::Insert { texts }) => {
            writeln!(buffer, "Inserting {} text(s).", texts.len())?;
            let insertion_results = insert_texts(texts)?;
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
