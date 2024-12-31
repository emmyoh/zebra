use bytes::Bytes;
use clap::{command, Parser, Subcommand};
use indicatif::HumanCount;
use indicatif::ProgressStyle;
use indicatif::{ProgressBar, ProgressDrawTarget};
use pretty_duration::pretty_duration;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rodio::{Decoder, OutputStream, Sink};
use space::Metric;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use ticky::Stopwatch;
use zebra::database::core::Database;
use zebra::database::default::audio::DefaultAudioDatabase;
use zebra::database::default::image::DefaultImageDatabase;
use zebra::database::default::text::DefaultTextDatabase;
use zebra::distance::DistanceUnit;
use zebra::model::core::DatabaseEmbeddingModel;
use zebra::Embedding;

#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help(true))]
struct Cli {
    #[structopt(subcommand)]
    commands: Commands,
    #[arg(short, long, global = true)]
    database_path: String,
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
    InsertFromFiles {
        file_paths: Vec<PathBuf>,
        #[arg(default_value_t = 100)]
        batch_size: usize,
    },
    #[command(about = "Query texts from the database.", arg_required_else_help(true))]
    Query {
        texts: Vec<String>,
        #[arg(default_value_t = 1)]
        number_of_results: usize,
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
    Insert {
        file_paths: Vec<PathBuf>,
        #[arg(default_value_t = 100)]
        batch_size: usize,
    },
    #[command(
        about = "Query images from the database.",
        arg_required_else_help(true)
    )]
    Query {
        image_path: PathBuf,
        #[arg(default_value_t = 1)]
        number_of_results: usize,
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
    Insert {
        file_paths: Vec<PathBuf>,
        #[arg(default_value_t = 100)]
        batch_size: usize,
    },
    #[command(
        about = "Query sounds from the database.",
        arg_required_else_help(true)
    )]
    Query {
        audio_path: PathBuf,
        #[arg(default_value_t = 1)]
        number_of_results: usize,
    },
    #[command(about = "Clear the database.")]
    Clear,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.commands {
        Commands::Text(text) => match text.text_commands {
            TextCommands::Insert { texts } => {
                let mut sw = Stopwatch::start_new();
                let mut db = DefaultTextDatabase::open_or_create(&cli.database_path);
                let mut buffer = BufWriter::new(stdout().lock());
                writeln!(buffer, "Inserting {} text(s).", texts.len())?;
                let texts_bytes: Vec<_> = texts.into_par_iter().map(|x| Bytes::from(x)).collect();
                let insertion_results = db.insert_documents(texts_bytes)?;
                sw.stop();
                writeln!(
                    buffer,
                    "{} embeddings of {} dimensions inserted into the database in {}.",
                    HumanCount(insertion_results.0.try_into()?).to_string(),
                    HumanCount(insertion_results.1.try_into()?).to_string(),
                    pretty_duration(&sw.elapsed(), None)
                )?;
            }
            TextCommands::InsertFromFiles {
                file_paths,
                batch_size,
            } => {
                let mut db = DefaultTextDatabase::open_or_create(&cli.database_path);
                insert_from_files(&mut db, file_paths, batch_size)?;
            }
            TextCommands::Query {
                texts,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = DefaultTextDatabase::open_or_create(&cli.database_path);
                let mut buffer = BufWriter::new(stdout().lock());
                let num_texts = texts.len();
                writeln!(buffer, "Querying {} text(s).", num_texts)?;
                let texts_bytes: Vec<_> = texts.into_par_iter().map(|x| Bytes::from(x)).collect();
                let query_results = db.query_documents(texts_bytes, number_of_results)?;
                let result_texts: Vec<_> = query_results
                    .iter()
                    .map(|x| String::from_utf8_lossy(x))
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
                DefaultTextDatabase::open_or_create(&cli.database_path).clear_database();
            }
        },
        Commands::Image(image) => match image.image_commands {
            ImageCommands::Insert {
                file_paths,
                batch_size,
            } => {
                let mut db = DefaultImageDatabase::open_or_create(&cli.database_path);
                insert_from_files(&mut db, file_paths, batch_size)?;
            }
            ImageCommands::Query {
                image_path,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = DefaultImageDatabase::open_or_create(&cli.database_path);
                let mut buffer = BufWriter::new(stdout().lock());
                let image_print_config = viuer::Config {
                    transparent: true,
                    premultiplied_alpha: false,
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
                writeln!(buffer, "Querying image.")?;
                let image_bytes = std::fs::read(image_path).unwrap_or_default().into();
                let query_results = db.query_documents(vec![image_bytes], number_of_results)?;
                sw.stop();
                writeln!(
                    buffer,
                    "Queried image in {}.",
                    pretty_duration(&sw.elapsed(), None)
                )?;
                writeln!(buffer, "Results:")?;
                for result in query_results {
                    let path = PathBuf::from(String::from_utf8(result)?);
                    let _ = viuer::print_from_file(&path, &image_print_config);
                }
            }
            ImageCommands::Clear => {
                DefaultImageDatabase::open_or_create(&cli.database_path).clear_database();
            }
        },
        Commands::Audio(audio) => match audio.audio_commands {
            AudioCommands::Insert {
                file_paths,
                batch_size,
            } => {
                let mut db = DefaultAudioDatabase::open_or_create(&cli.database_path);
                insert_from_files(&mut db, file_paths, batch_size)?;
            }
            AudioCommands::Query {
                audio_path,
                number_of_results,
            } => {
                let mut sw = Stopwatch::start_new();
                let mut db = DefaultAudioDatabase::open_or_create(&cli.database_path);
                let (_stream, stream_handle) = OutputStream::try_default()?;
                let sink = Sink::try_new(&stream_handle)?;
                let mut buffer = BufWriter::new(stdout().lock());
                writeln!(buffer, "Querying sound.")?;
                let audio_bytes = std::fs::read(audio_path).unwrap_or_default().into();
                let query_results = db.query_documents(vec![audio_bytes], number_of_results)?;
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
                DefaultAudioDatabase::open_or_create(&cli.database_path).clear_database();
            }
        },
    }
    Ok(())
}

fn progress_bar_style() -> anyhow::Result<ProgressStyle> {
    Ok(ProgressStyle::with_template("[{elapsed} elapsed, {eta} remaining ({duration} total)] {wide_bar:.cyan/blue} {human_pos} of {human_len} ({percent}%) {msg}")?)
}

fn insert_from_files<
    Met: Metric<Embedding, Unit = DistanceUnit> + Default + serde::ser::Serialize,
    Mod: DatabaseEmbeddingModel + Default + serde::ser::Serialize,
    const EF_CONSTRUCTION: usize,
    const M: usize,
    const M0: usize,
>(
    db: &mut Database<Met, Mod, EF_CONSTRUCTION, M, M0>,
    file_paths: Vec<PathBuf>,
    batch_size: usize,
) -> anyhow::Result<()>
where
    for<'de> Met: serde::Deserialize<'de>,
    for<'de> Mod: serde::Deserialize<'de>,
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
    let documents: Vec<_> = file_paths
        .par_iter()
        .filter_map(|x| std::fs::read(x).ok().map(|y| y.into()))
        .collect();
    // Insert documents in batches.
    for document_batch in documents.chunks(batch_size) {
        let mut batch_sw = Stopwatch::start_new();
        let insertion_results = db.insert_documents(document_batch.to_vec())?;
        batch_sw.stop();
        progress_bar.println(format!(
            "{} embeddings of {} dimensions inserted into the database in {}.",
            HumanCount(insertion_results.0 as u64).to_string(),
            HumanCount(insertion_results.1 as u64).to_string(),
            pretty_duration(&batch_sw.elapsed(), None)
        ));
        progress_bar.inc(batch_size as u64);
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
