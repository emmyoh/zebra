use crate::image::load_image224;
use bytes::Bytes;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::vit;
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use sonogram::ColourGradient;
use sonogram::FrequencyScale;
use sonogram::SpecOptionsBuilder;
use std::error::Error;
use std::io::Cursor;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// A trait for embedding models that can be used with the database.
pub trait DatabaseEmbeddingModel {
    /// Create a new instance of the embedding model.
    fn new() -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    /// Embed a vector of documents.
    ///
    /// # Arguments
    ///
    /// * `documents` - A vector of documents to be embedded.
    ///
    /// # Returns
    ///
    /// A vector of embeddings.
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>>;

    /// Embed a single document.
    ///
    /// # Arguments
    ///
    /// * `document` – A single document to be embedded.
    ///
    /// # Returns
    ///
    /// An embedding vector.
    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>>;
}

impl DatabaseEmbeddingModel for TextEmbedding {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )?)
    }
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>> {
        Ok(self.embed(
            documents
                .into_par_iter()
                .map(|x| x.to_vec())
                .filter_map(|x| String::from_utf8(x).ok())
                .collect(),
            None,
        )?)
    }

    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>> {
        let vec_with_document = vec![document]
            .into_par_iter()
            .map(|x| x.to_vec())
            .filter_map(|x| String::from_utf8(x).ok())
            .collect();
        let vector_of_embeddings = self.embed(vec_with_document, None)?;
        Ok(vector_of_embeddings.first().unwrap().to_vec())
    }
}

/// A model for embedding images.
pub struct ImageEmbeddingModel;

impl DatabaseEmbeddingModel for ImageEmbeddingModel {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self)
    }
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>> {
        let mut result = Vec::new();
        let device = candle_examples::device(false)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(
            &vit::Config::vit_base_patch16_224(),
            false,
            varbuilder.pp("vit").pp("embeddings"),
        )?;
        for document in documents {
            let image = load_image224(document)?.to_device(&device)?;
            let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
            let embedding_vector = embedding_tensors.flatten_all()?.to_vec1::<f32>()?;
            result.push(embedding_vector);
        }
        Ok(result)
    }
    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>> {
        let device = candle_examples::device(false)?;
        let image = load_image224(document)?.to_device(&device)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(&vit::Config::vit_base_patch16_224(), false, varbuilder)?;
        let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
        let embedding_vector = embedding_tensors.to_vec1::<f32>()?;
        Ok(embedding_vector)
    }
}

/// A model for embedding audio.
pub struct AudioEmbeddingModel;

impl AudioEmbeddingModel {
    /// Decodes the samples of an audio files.
    ///
    /// # Arguments
    ///
    /// * `audio` - The raw bytes of an audio file.
    ///
    /// # Returns
    ///
    /// An `i16` vector of decoded samples, and the sample rate of the audio.
    pub fn audio_to_data(audio: Bytes) -> Result<(Vec<i16>, u32), Box<dyn Error>> {
        let mss = MediaSourceStream::new(Box::new(Cursor::new(audio)), Default::default());
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        let probed =
            symphonia::default::get_probe().format(&Hint::new(), mss, &fmt_opts, &meta_opts)?;
        let mut format = probed.format;
        let track = format
            .tracks()
            .into_par_iter()
            .find_any(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .unwrap();
        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;
        let track_id = track.id;
        let mut sample_rate = 0;
        let mut data = Vec::new();

        loop {
            match format.next_packet() {
                Ok(packet) => {
                    while !format.metadata().is_latest() {
                        format.metadata().pop();
                    }
                    if packet.track_id() != track_id {
                        continue;
                    }
                    match decoder.decode(&packet) {
                        Ok(decoded) => {
                            let decoded = decoded.make_equivalent::<i16>();
                            sample_rate = decoded.spec().rate;
                            let number_channels = decoded.spec().channels.count();
                            for i in 0..number_channels {
                                let samples = decoded.chan(i);
                                data.extend_from_slice(samples);
                            }
                        }
                        Err(_) => continue,
                    }
                }
                Err(_) => break,
            }
        }

        Ok((data, sample_rate))
    }

    /// Convert an audio file into a logarithm-scale spectrogram for use with image embedding models.
    ///
    /// # Arguments
    ///
    /// `audio` - The raw bytes of an audio file.
    ///
    /// # Returns
    ///
    /// A spectrogram of the audio as an ImageNet-normalised tensor with shape [3 224 224].
    pub fn audio_to_image_tensor(audio: Bytes) -> Result<Tensor, Box<dyn Error>> {
        let (data, sample_rate) = Self::audio_to_data(audio)?;
        let mut spectrograph = SpecOptionsBuilder::new(512)
            .load_data_from_memory(data, sample_rate)
            .normalise()
            .build()
            .unwrap();
        let mut spectrogram = spectrograph.compute();
        let mut gradient = ColourGradient::rainbow_theme();
        let png_bytes =
            spectrogram.to_png_in_memory(FrequencyScale::Log, &mut gradient, 224, 224)?;
        let img = image::load_from_memory_with_format(&png_bytes, image::ImageFormat::Png)
            .map_err(candle_core::Error::wrap)?
            .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
        Ok((data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?)
    }
}

impl DatabaseEmbeddingModel for AudioEmbeddingModel {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self)
    }
    fn embed_documents(&self, documents: Vec<Bytes>) -> Result<Vec<Embedding>, Box<dyn Error>> {
        let mut result = Vec::new();
        let device = candle_examples::device(false)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(
            &vit::Config::vit_base_patch16_224(),
            false,
            varbuilder.pp("vit").pp("embeddings"),
        )?;
        for document in documents {
            let image = AudioEmbeddingModel::audio_to_image_tensor(document)?.to_device(&device)?;
            let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
            let embedding_vector = embedding_tensors.flatten_all()?.to_vec1::<f32>()?;
            result.push(embedding_vector);
        }
        Ok(result)
    }
    fn embed(&self, document: Bytes) -> Result<Embedding, Box<dyn Error>> {
        let device = candle_examples::device(false)?;
        let image = AudioEmbeddingModel::audio_to_image_tensor(document)?.to_device(&device)?;
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("google/vit-base-patch16-224".into());
        let model_file = api.get("model.safetensors")?;
        let varbuilder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = vit::Embeddings::new(&vit::Config::vit_base_patch16_224(), false, varbuilder)?;
        let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
        let embedding_vector = embedding_tensors.to_vec1::<f32>()?;
        Ok(embedding_vector)
    }
}
