use clap::{command, Parser, Subcommand};
use fastembed::TextEmbedding;
use indicatif::{ProgressBar, ProgressDrawTarget};
use pretty_duration::pretty_duration;
use rodio::{Decoder, OutputStream, Sink};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::io::{stdout, BufWriter};
use std::path::PathBuf;
use ticky::Stopwatch;
use zebra::db::DocumentType;
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
    Query { texts: Vec<String> },
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
    Query { image_path: PathBuf },
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
    Query { audio_path: PathBuf },
    #[command(about = "Clear the database.")]
    Clear,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let mut sw = Stopwatch::start_new();
    match cli.commands {
        Commands::Text(text) => {
            match text.text_commands {
                TextCommands::Insert { mut texts } => {
                    let mut db = zebra::text::create_or_load_database()?;
                    let mut buffer = BufWriter::new(stdout().lock());
                    let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                    writeln!(buffer, "Inserting {} text(s).", texts.len())?;
                    let insertion_results = db.insert_documents(&model, &mut texts)?;
                    sw.stop();
                    writeln!(
                        buffer,
                        "{} embeddings of {} dimensions inserted into the database in {}.",
                        insertion_results.0,
                        insertion_results.1,
                        pretty_duration(&sw.elapsed(), None)
                    )?;
                }
                TextCommands::InsertFromFiles { file_paths } => {
                    let mut db = zebra::text::create_or_load_database()?;
                    let num_texts = file_paths.len();
                    let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                    println!("Inserting texts from {} file(s).", num_texts);
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
                            let insertion_results = db.insert_documents(&model, &mut texts)?;
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
                        let insertion_results = db.insert_documents(&model, &mut texts)?;
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
                TextCommands::Query { texts } => {
                    let mut db = zebra::text::create_or_load_database()?;
                    let mut buffer = BufWriter::new(stdout().lock());
                    let num_texts = texts.len();
                    let model: TextEmbedding = DatabaseEmbeddingModel::new()?;
                    writeln!(buffer, "Querying {} text(s).", num_texts)?;
                    let query_results = db.query_documents(&model, texts)?;
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
                    let text_type = DocumentType::Text;
                    let mut buffer = BufWriter::new(stdout().lock());
                    writeln!(buffer, "Clearing database.")?;
                    std::fs::remove_file(text_type.database_name()).unwrap_or(());
                    std::fs::remove_dir_all(text_type.subdirectory_name()).unwrap_or(());
                    std::fs::remove_dir_all(".fastembed_cache").unwrap_or(());
                    sw.stop();
                    writeln!(
                        buffer,
                        "Database cleared in {}.",
                        pretty_duration(&sw.elapsed(), None)
                    )?;
                }
            }
        }
        Commands::Image(image) => match image.image_commands {
            ImageCommands::Insert { file_paths } => {
                let mut db = zebra::image::create_or_load_database()?;
                let num_images = file_paths.len();
                let model: ImageEmbeddingModel = DatabaseEmbeddingModel::new()?;
                println!("Inserting images from {} file(s).", num_images);
                let progress_bar = ProgressBar::with_draw_target(
                    Some(num_images.try_into()?),
                    ProgressDrawTarget::hidden(),
                )
                .with_message(format!("Inserting images from {} file(s).", num_images));
                let images: Vec<String> = file_paths
                    .into_iter()
                    .map(|x| x.to_str().unwrap().to_string())
                    .collect();
                // Insert images in batches of INSERT_BATCH_SIZE.
                for image_batch in images.chunks(INSERT_BATCH_SIZE) {
                    let insertion_results = db.insert_documents(&model, image_batch)?;
                    progress_bar.println(format!(
                        "{} embeddings of {} dimensions inserted into the database.",
                        insertion_results.0, insertion_results.1
                    ));
                    progress_bar.inc(INSERT_BATCH_SIZE.try_into()?);
                    if progress_bar.is_hidden() {
                        progress_bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(100));
                    }
                }
                sw.stop();
                progress_bar.println(format!(
                    "Inserted {} image(s) in {}.",
                    num_images,
                    pretty_duration(&sw.elapsed(), None)
                ));
            }
            ImageCommands::Query { image_path } => {
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
                let query_results =
                    db.query_documents(&model, vec![image_path.to_str().unwrap()])?;
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
                let image_type = DocumentType::Image;
                let mut buffer = BufWriter::new(stdout().lock());
                writeln!(buffer, "Clearing database.")?;
                std::fs::remove_file(image_type.database_name()).unwrap_or(());
                std::fs::remove_dir_all(image_type.subdirectory_name()).unwrap_or(());
                // std::fs::remove_dir_all(".fastembed_cache").unwrap_or(());
                sw.stop();
                writeln!(
                    buffer,
                    "Database cleared in {}.",
                    pretty_duration(&sw.elapsed(), None)
                )?;
            }
        },
        Commands::Audio(audio) => match audio.audio_commands {
            AudioCommands::Insert { file_paths } => {
                let mut db = zebra::audio::create_or_load_database()?;
                let num_sounds = file_paths.len();
                let model: AudioEmbeddingModel = DatabaseEmbeddingModel::new()?;
                println!("Inserting sounds from {} file(s).", num_sounds);
                let progress_bar = ProgressBar::with_draw_target(
                    Some(num_sounds.try_into()?),
                    ProgressDrawTarget::hidden(),
                )
                .with_message(format!("Inserting sounds from {} file(s).", num_sounds));
                let sounds: Vec<String> = file_paths
                    .into_iter()
                    .map(|x| x.to_str().unwrap().to_string())
                    .collect();
                // Insert sounds in batches of INSERT_BATCH_SIZE.
                for image_batch in sounds.chunks(INSERT_BATCH_SIZE) {
                    let insertion_results = db.insert_documents(&model, image_batch)?;
                    progress_bar.println(format!(
                        "{} embeddings of {} dimensions inserted into the database.",
                        insertion_results.0, insertion_results.1
                    ));
                    progress_bar.inc(INSERT_BATCH_SIZE.try_into()?);
                    if progress_bar.is_hidden() {
                        progress_bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(100));
                    }
                }
                sw.stop();
                progress_bar.println(format!(
                    "Inserted {} sound(s) in {}.",
                    num_sounds,
                    pretty_duration(&sw.elapsed(), None)
                ));
            }
            AudioCommands::Query { audio_path } => {
                let mut db = zebra::audio::create_or_load_database()?;
                let (_stream, stream_handle) = OutputStream::try_default()?;
                let sink = Sink::try_new(&stream_handle)?;
                let mut buffer = BufWriter::new(stdout().lock());
                let model: AudioEmbeddingModel = DatabaseEmbeddingModel::new()?;
                writeln!(buffer, "Querying sound.")?;
                let query_results =
                    db.query_documents(&model, vec![audio_path.to_str().unwrap()])?;
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
                let audio_type = DocumentType::Audio;
                let mut buffer = BufWriter::new(stdout().lock());
                writeln!(buffer, "Clearing database.")?;
                std::fs::remove_file(audio_type.database_name()).unwrap_or(());
                std::fs::remove_dir_all(audio_type.subdirectory_name()).unwrap_or(());
                // std::fs::remove_dir_all(".fastembed_cache").unwrap_or(());
                sw.stop();
                writeln!(
                    buffer,
                    "Database cleared in {}.",
                    pretty_duration(&sw.elapsed(), None)
                )?;
            }
        },
        // _ => unreachable!(),
    }
    Ok(())
}
