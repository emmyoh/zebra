use super::{
    core::{DatabaseEmbeddingModel, DIM_VIT_BASE_PATCH16_224},
    image::ImageEmbeddingModel,
};
use crate::Embedding;
use anyhow::anyhow;
use bitcode::{Decode, Encode};
use bytes::Bytes;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::vit;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sonogram::{ColourGradient, FrequencyScale, SpecOptionsBuilder};
use std::io::Cursor;
use symphonia::core::{
    audio::Signal,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

/// A trait for audio embedding models; these models are a subset of image embedding models.
pub trait AudioEmbeddingModel: ImageEmbeddingModel {
    /// Decodes the samples of an audio files.
    ///
    /// # Arguments
    ///
    /// * `audio` - The raw bytes of an audio file.
    ///
    /// # Returns
    ///
    /// An `i16` vector of decoded samples, and the sample rate of the audio.
    fn audio_to_data(&self, audio: &Bytes) -> anyhow::Result<(Vec<i16>, u32)> {
        let mss = MediaSourceStream::new(Box::new(Cursor::new(audio.to_vec())), Default::default());
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        let probed =
            symphonia::default::get_probe().format(&Hint::new(), mss, &fmt_opts, &meta_opts)?;
        let mut format = probed.format;
        let track = format
            .tracks()
            .into_par_iter()
            .find_any(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(anyhow!("No tracks found in audio … "))?;
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
    fn audio_to_image_tensor224(&self, audio: &Bytes) -> anyhow::Result<Tensor> {
        let (data, sample_rate) = self.audio_to_data(audio)?;
        let mut spectrograph = SpecOptionsBuilder::new(512)
            .load_data_from_memory(data, sample_rate)
            .normalise()
            .build()
            .ok()
            .ok_or(anyhow!("Unable to compute spectrograph … "))?;
        let mut spectrogram = spectrograph.compute();
        let mut gradient = ColourGradient::rainbow_theme();
        let png_bytes =
            spectrogram.to_png_in_memory(FrequencyScale::Log, &mut gradient, 224, 224)?;
        self.load_image224(&Bytes::from(png_bytes))
    }
}

/// A model for embedding audio.
#[derive(Default, Encode, Decode, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VitBasePatch16_224;
impl ImageEmbeddingModel for VitBasePatch16_224 {}
impl AudioEmbeddingModel for VitBasePatch16_224 {}

impl DatabaseEmbeddingModel<DIM_VIT_BASE_PATCH16_224> for VitBasePatch16_224 {
    fn embed_documents(
        &self,
        documents: &Vec<Bytes>,
    ) -> anyhow::Result<Vec<Embedding<DIM_VIT_BASE_PATCH16_224>>> {
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
            let image = self
                .audio_to_image_tensor224(document)?
                .to_device(&device)?;
            let embedding_tensors = model.forward(&image.unsqueeze(0)?, None, false)?;
            let embedding_vector = embedding_tensors.flatten_all()?.to_vec1::<f32>()?;
            result.push(embedding_vector);
        }
        Ok(result
            .into_par_iter()
            .map(|x| x.try_into().unwrap_or([0.0; DIM_VIT_BASE_PATCH16_224]))
            .collect())
    }
}
