use clap::{command, Parser, Subcommand};
use fastembed::Embedding;
use fastembed::TextEmbedding;
use indicatif::HumanCount;
use indicatif::ProgressStyle;
use indicatif::{ProgressBar, ProgressDrawTarget};
use pretty_duration::pretty_duration;
use rodio::{Decoder, OutputStream, Sink};
use space::Metric;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use ticky::Stopwatch;
use zebra::db::Database;
use zebra::db::DocumentType;
use zebra::distance::DistanceUnit;
use zebra::model::{AudioEmbeddingModel, DatabaseEmbeddingModel, ImageEmbeddingModel};

const INSERT_BATCH_SIZE: usize = 100;

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help(true))]
struct Cli {
    #[structopt(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[clap(about = "Text commands.")]
    Text(Text),
    #[clap(about = "Image commands.")]
    Image(Image),
    #[clap(about = "Audio commands.")]
    Audio(Audio),
}

#[derive(Parser)]
struct Text {
    #[structopt(subcommand)]
    text_commands: TextCommands,
}

#[derive(Parser)]
struct Image {
    #[structopt(subcommand)]
    image_commands: ImageCommands,
}

#[derive(Parser)]
struct Audio {
    #[structopt(subcommand)]
    audio_commands: AudioCommands,
}

#[derive(Subcommand)]
enum TextCommands {
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
    Query {
        texts: Vec<String>,
        number_of_results: Option<usize>,
    },
    #[command(about = "Clear the database.")]
    Clear,
}

#[derive(Subcommand)]
enum ImageCommands {
    #[command(
        about = "Insert images into the database.",
        arg_required_else_help(true)
    )]
    Insert { file_paths: Vec<PathBuf> },
    #[command(
        about = "Query images from the database.",
        arg_required_else_help(true)
    )]
    Query {
        image_path: PathBuf,
        number_of_results: Option<usize>,
    },
    #[command(about = "Clear the database.")]
    Clear,
}

#[derive(Subcommand)]
enum AudioCommands {
    #[command(
        about = "Insert sounds into the database.",
        arg_required_else_help(true)
    )]
    Insert { file_paths: Vec<PathBuf> },
    #[command(
        about = "Query sounds from the database.",
        arg_required_else_help(true)
    )]
    Query {
        audio_path: PathBuf,
        number_of_results: Option<usize>,
    },
    #[command(about = "Clear the database.")]
    Clear,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.commands {
        Commands::Text(text) => match text.text_commands {
            TextCommands::Insert { mut texts } => {
                let mut sw = Stopwatch::start_new();
                let mut db = zebra::text::create_or_load_database()?;
                let mut buffer = BufWriter::new(stdout().lock());
                let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                writeln!(buffer, "Inserting {} text(s).", texts.len())?;
                let insertion_results = db.insert_documents(&model, &mut texts)?;
                sw.stop();
                writeln!(
                    buffer,
                    "{} embeddings of {} dimensions inserted into the database in {}.",
                    HumanCount(insertion_results.0.try_into()?).to_string(),
                    HumanCount(insertion_results.1.try_into()?).to_string(),
                    pretty_duration(&sw.elapsed(), None)
                )?;
            }
            TextCommands::InsertFromFiles { file_paths } => {
                let mut db = zebra::text::create_or_load_database()?;
                let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                insert_from_files(&mut db, model, file_paths)?;
            }
            TextCommands::Query {
                texts,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = zebra::text::create_or_load_database()?;
                let mut buffer = BufWriter::new(stdout().lock());
                let num_texts = texts.len();
                let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                writeln!(buffer, "Querying {} text(s).", num_texts)?;
                let query_results = db.query_documents(&model, texts, number_of_results)?;
                let result_texts: Vec<String> = query_results
                    .iter()
                    .map(|x| String::from_utf8(x.to_vec()).unwrap())
                    .collect();
                sw.stop();
                writeln!(
                    buffer,
                    "Queried {} text(s) in {}.",
                    num_texts,
                    pretty_duration(&sw.elapsed(), None)
                )?;
                writeln!(buffer, "Results:")?;
                for result in result_texts {
                    writeln!(buffer, "- \t{}", result)?;
                }
            }
            TextCommands::Clear => {
                clear_database(DocumentType::Text)?;
            }
        },
        Commands::Image(image) => match image.image_commands {
            ImageCommands::Insert { file_paths } => {
                let mut db = zebra::image::create_or_load_database()?;
                let model: ImageEmbeddingModel = DatabaseEmbeddingModel::new()?;
                insert_from_files(&mut db, model, file_paths)?;
            }
            ImageCommands::Query {
                image_path,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = zebra::image::create_or_load_database()?;
                let mut buffer = BufWriter::new(stdout().lock());
                let image_print_config = viuer::Config {
                    transparent: true,
                    absolute_offset: false,
                    x: 0,
                    y: 0,
                    restore_cursor: true,
                    width: None,
                    height: None,
                    truecolor: true,
                    use_kitty: true,
                    use_iterm: true,
                    #[cfg(feature = "sixel")]
                    use_sixel: true,
                };
                let model: ImageEmbeddingModel = DatabaseEmbeddingModel::new()?;
                writeln!(buffer, "Querying image.")?;
                let query_results = db.query_documents(
                    &model,
                    vec![image_path.to_str().unwrap()],
                    number_of_results,
                )?;
                sw.stop();
                writeln!(
                    buffer,
                    "Queried image in {}.",
                    pretty_duration(&sw.elapsed(), None)
                )?;
                writeln!(buffer, "Results:")?;
                for result in query_results {
                    let path = PathBuf::from(String::from_utf8(result)?);
                    let _print_result = viuer::print_from_file(&path, &image_print_config);
                }
            }
            ImageCommands::Clear => {
                clear_database(DocumentType::Image)?;
            }
        },
        Commands::Audio(audio) => match audio.audio_commands {
            AudioCommands::Insert { file_paths } => {
                let mut db = zebra::audio::create_or_load_database()?;
                let model: AudioEmbeddingModel = DatabaseEmbeddingModel::new()?;
                insert_from_files(&mut db, model, file_paths)?;
            }
            AudioCommands::Query {
                audio_path,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = zebra::audio::create_or_load_database()?;
                let (_stream, stream_handle) = OutputStream::try_default()?;
                let sink = Sink::try_new(&stream_handle)?;
                let mut buffer = BufWriter::new(stdout().lock());
                let model: AudioEmbeddingModel = DatabaseEmbeddingModel::new()?;
                writeln!(buffer, "Querying sound.")?;
                let query_results = db.query_documents(
                    &model,
                    vec![audio_path.to_str().unwrap()],
                    number_of_results,
                )?;
                sw.stop();
                writeln!(
                    buffer,
                    "Queried sound in {}.",
                    pretty_duration(&sw.elapsed(), None)
                )?;
                writeln!(buffer, "Results:")?;
                for result in query_results {
                    let path = PathBuf::from(String::from_utf8(result)?);
                    writeln!(buffer, "Playing {} … ", path.to_string_lossy())?;
                    let file = BufReader::new(File::open(path)?);
                    let source = Decoder::new(file)?;
                    sink.append(source);
                    sink.sleep_until_end();
                }
            }
            AudioCommands::Clear => {
                clear_database(DocumentType::Audio)?;
            }
        },
        // _ => unreachable!(),
    }
    Ok(())
}

fn progress_bar_style() -> Result<ProgressStyle, Box<dyn Error>> {
    Ok(ProgressStyle::with_template("[{elapsed} elapsed, {eta} remaining ({duration} total)] {wide_bar:.cyan/blue} {human_pos} of {human_len} ({percent}%) {msg}")?)
}

fn clear_database(document_type: DocumentType) -> Result<(), Box<dyn Error>> {
    let mut sw = Stopwatch::start_new();
    let mut buffer = BufWriter::new(stdout().lock());
    writeln!(buffer, "Clearing database.")?;
    std::fs::remove_file(document_type.database_name()).unwrap_or(());
    std::fs::remove_dir_all(document_type.subdirectory_name()).unwrap_or(());
    if document_type == DocumentType::Text {
        std::fs::remove_dir_all(".fastembed_cache").unwrap_or(());
    }
    sw.stop();
    writeln!(
        buffer,
        "Database cleared in {}.",
        pretty_duration(&sw.elapsed(), None)
    )?;
    Ok(())
}

fn insert_from_files<
    Met: Metric<Embedding, Unit = DistanceUnit> + serde::ser::Serialize,
    const EF_CONSTRUCTION: usize,
    const M: usize,
    const M0: usize,
>(
    db: &mut Database<Met, EF_CONSTRUCTION, M, M0>,
    model: impl DatabaseEmbeddingModel,
    file_paths: Vec<PathBuf>,
) -> Result<(), Box<dyn Error>>
where
    for<'de> Met: serde::Deserialize<'de>,
{
    let mut sw = Stopwatch::start_new();
    let num_documents = file_paths.len();
    println!(
        "Inserting documents from {} file(s).",
        HumanCount(num_documents.try_into()?).to_string()
    );
    let progress_bar = ProgressBar::with_draw_target(
        Some(num_documents.try_into()?),
        ProgressDrawTarget::hidden(),
    );
    progress_bar.set_style(progress_bar_style()?);
    let documents: Vec<String> = file_paths
        .into_iter()
        .map(|x| x.to_str().unwrap().to_string())
        .collect();
    // Insert documents in batches of INSERT_BATCH_SIZE.
    for document_batch in documents.chunks(INSERT_BATCH_SIZE) {
        let mut batch_sw = Stopwatch::start_new();
        let insertion_results = db.insert_documents(&model, document_batch)?;
        batch_sw.stop();
        progress_bar.println(format!(
            "{} embeddings of {} dimensions inserted into the database in {}.",
            HumanCount(insertion_results.0.try_into()?).to_string(),
            HumanCount(insertion_results.1.try_into()?).to_string(),
            pretty_duration(&batch_sw.elapsed(), None)
        ));
        progress_bar.inc(INSERT_BATCH_SIZE.try_into()?);
        if progress_bar.is_hidden() {
            progress_bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(100));
        }
    }
    sw.stop();
    progress_bar.println(format!(
        "Inserted {} document(s) in {}.",
        num_documents,
        pretty_duration(&sw.elapsed(), None)
    ));
    Ok(())
}
