use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use hound::{SampleFormat, WavReader};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::dictation::resample_for_parakeet;
use crate::dictation::vocabulary::{self, VocabularyBoost, VocabularyEntry, bias_for};
use crate::parakeet::Parakeet;
use crate::transcription::gguf_session::RunBias;
use crate::transcription::models::{TranscriptionModelId, VocabularyUse, catalog_match};
use crate::transcription::pad_gguf_samples;

pub enum Backend {
    Onnx,
    TranscribeCpp {
        model: PathBuf,
        /// The catalog model the file is (name and size), resolved once, so
        /// clips are prepared as dictation prepares them for it. `None` for a
        /// file that is not a catalog artifact.
        model_id: Option<TranscriptionModelId>,
        /// `--vocabulary`: every clip is biased with it, as dictation biases.
        vocabulary: Option<RunVocabulary>,
    },
}

impl Backend {
    /// The transcribe-cpp backend for `model`, reading `--vocabulary` when
    /// given. Only a Parakeet model takes the bias: dictation hands Whisper and
    /// Qwen3-ASR the list as a prompt, which a bias would not measure.
    pub fn transcribe_cpp(
        model: PathBuf,
        vocabulary: Option<&Path>,
        boost: Option<VocabularyBoost>,
    ) -> std::result::Result<Self, String> {
        let vocabulary = vocabulary
            .map(|path| {
                RunVocabulary::read(path, boost.unwrap_or_default())
                    .map_err(|error| format!("--vocabulary {error}"))
            })
            .transpose()?;
        let model_id = model
            .file_name()
            .and_then(|name| name.to_str())
            .zip(fs::metadata(&model).ok())
            .and_then(|(name, meta)| catalog_match(None, None, name, meta.len()));
        if vocabulary.is_some() {
            require_bias_model(model_id)?;
        }
        Ok(Self::TranscribeCpp {
            model,
            model_id,
            vocabulary,
        })
    }

    /// The catalog model the clips are prepared for.
    fn model_id(&self) -> Option<TranscriptionModelId> {
        match self {
            Self::TranscribeCpp { model_id, .. } => *model_id,
            Self::Onnx => None,
        }
    }

    /// Said on stderr when an existing file is no catalog artifact, so a
    /// renamed copy cannot silently lose the model's trailing silence.
    fn unrecognised_model_note(&self) -> Option<&'static str> {
        match self {
            Self::TranscribeCpp {
                model,
                model_id: None,
                ..
            } if model.is_file() => Some(
                "note: the model file is not a catalog artifact (by name and size), so clips get no model trailing silence",
            ),
            _ => None,
        }
    }

    /// The `model` record's `bias_vocabulary`, only with `--vocabulary`.
    fn bias_record(&self) -> Option<serde_json::Value> {
        match self {
            Self::TranscribeCpp {
                vocabulary: Some(vocabulary),
                ..
            } => Some(vocabulary.record()),
            _ => None,
        }
    }
}

/// An early failure for the catalog models that never take a bias, before
/// anything loads. The load checks every file again (`OfflineGgufSession::fix_run_bias`), which
/// also covers one this cannot name.
fn require_bias_model(id: Option<TranscriptionModelId>) -> std::result::Result<(), String> {
    let Some(id) = id else { return Ok(()) };
    let definition = crate::transcription::models::definition(id);
    if definition.vocabulary_use() == VocabularyUse::Bias {
        return Ok(());
    }
    let why = match definition.vocabulary_use() {
        VocabularyUse::Prompt => {
            "reads its vocabulary as a context prompt, which a bias does not measure"
        }
        _ => "takes no vocabulary",
    };
    Err(format!(
        "--vocabulary biases only Parakeet models, and {} {why}",
        definition.name
    ))
}

/// The terms and boost every run is biased with, as dictation builds them
/// (same cleaning, limits, and 64-term cut).
pub struct RunVocabulary {
    bias: RunBias,
    /// Entries read, before the engine's term cut.
    input_terms: usize,
    boost: VocabularyBoost,
}

impl RunVocabulary {
    /// Reads a `--vocabulary` file: a JSON list in `settings.json`'s shape,
    /// each item a term string or `{"term", "sounds_like": []}`, held to the
    /// rules the Vocabulary pane applies. The error names the item.
    pub fn read(path: &Path, boost: VocabularyBoost) -> std::result::Result<Self, String> {
        let data = fs::read(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::parse(&data, boost).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn parse(data: &[u8], boost: VocabularyBoost) -> std::result::Result<Self, String> {
        let mut deserializer = serde_json::Deserializer::from_slice(data);
        let saved: Vec<VocabularyEntry> = serde_path_to_error::deserialize(&mut deserializer)
            .map_err(|error| format!("at {}: {}", error.path(), error.inner()))?;
        let mut entries = Vec::new();
        for (index, entry) in saved.iter().enumerate() {
            vocabulary::add(&mut entries, &entry.term, &entry.sounds_like)
                .map_err(|error| format!("at [{index}]: {error}"))?;
        }
        let bias = bias_for(&entries, boost).ok_or("the vocabulary has no terms")?;
        Ok(Self {
            bias,
            input_terms: entries.len(),
            boost,
        })
    }

    /// What the model record says was applied.
    fn record(&self) -> serde_json::Value {
        json!({
            "terms": self.bias.0.len(),
            "input_terms": self.input_terms,
            "boost": self.boost,
        })
    }
}

enum LoadedBackend {
    TranscribeCpp(Box<Parakeet>),
    #[cfg(feature = "onnx-benchmark")]
    Onnx(transcribe_rs::onnx::parakeet::ParakeetModel),
}

impl LoadedBackend {
    fn load(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Onnx => load_onnx(),
            Backend::TranscribeCpp {
                model, vocabulary, ..
            } => Ok(Self::TranscribeCpp(Box::new(Parakeet::load_for_benchmark(
                &model,
                vocabulary.map(|vocabulary| vocabulary.bias),
            )?))),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::TranscribeCpp(model) => model.name(),
            #[cfg(feature = "onnx-benchmark")]
            Self::Onnx(_) => "transcribe-rs-onnx-int8",
        }
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        match self {
            Self::TranscribeCpp(model) => model.transcribe(samples),
            #[cfg(feature = "onnx-benchmark")]
            Self::Onnx(model) => {
                use transcribe_rs::onnx::parakeet::{ParakeetParams, TimestampGranularity};

                model
                    .transcribe_with(
                        samples,
                        &ParakeetParams {
                            timestamp_granularity: Some(TimestampGranularity::Segment),
                            ..Default::default()
                        },
                    )
                    .map(|result| result.text)
                    .map_err(Into::into)
            }
        }
    }
}

#[cfg(feature = "onnx-benchmark")]
fn load_onnx() -> Result<LoadedBackend> {
    use transcribe_rs::onnx::Quantization;
    use transcribe_rs::onnx::parakeet::ParakeetModel;

    let model_path = crate::app_paths::models_dir()?.join("parakeet-tdt-0.6b-v2-int8");
    Ok(LoadedBackend::Onnx(
        ParakeetModel::load(&model_path, &Quantization::Int8).wrap_err_with(|| {
            format!(
                "could not load ONNX Parakeet v2 from {}",
                model_path.display()
            )
        })?,
    ))
}

#[cfg(not(feature = "onnx-benchmark"))]
fn load_onnx() -> Result<LoadedBackend> {
    Err(eyre!("the ONNX backend requires --features onnx-benchmark"))
}

#[derive(Deserialize)]
struct Manifest {
    /// Scoring vocabulary, not recognition hints. Never passed to the model.
    #[serde(default)]
    vocabulary: Vec<String>,
    clips: Vec<ClipDefinition>,
}

#[derive(Deserialize)]
struct ClipDefinition {
    id: String,
    audio: PathBuf,
    expected: String,
    #[serde(default)]
    expected_terms: Vec<String>,
    #[serde(default = "ordinary_group")]
    group: String,
}

fn ordinary_group() -> String {
    "ordinary".into()
}

struct Clip {
    id: String,
    expected: String,
    samples: Vec<f32>,
    audio_ms: f64,
    expected_terms: Vec<String>,
    group: String,
}

#[derive(Default, Serialize)]
struct Quality {
    clips: usize,
    word_errors: usize,
    reference_words: usize,
    expected_terms: usize,
    recalled_terms: usize,
    false_replacements: usize,
    clips_with_false_replacements: usize,
}

impl Quality {
    fn record(&mut self, clip: &Clip, actual: &str, vocabulary: &[String]) {
        let (errors, words) = word_errors(&clip.expected, actual);
        self.clips += 1;
        self.word_errors += errors;
        self.reference_words += words;
        self.expected_terms += clip.expected_terms.len();
        self.recalled_terms += clip
            .expected_terms
            .iter()
            .filter(|term| term_count(actual, term) > 0)
            .count();
        let false_replacements = vocabulary
            .iter()
            .map(|term| term_count(actual, term).saturating_sub(term_count(&clip.expected, term)))
            .sum::<usize>();
        self.false_replacements += false_replacements;
        self.clips_with_false_replacements += usize::from(false_replacements > 0);
    }

    fn report(&self) -> serde_json::Value {
        let mut report = serde_json::to_value(self).expect("quality counters serialize");
        report["wer"] = json!(ratio(self.word_errors, self.reference_words));
        report["term_recall"] = json!(ratio(self.recalled_terms, self.expected_terms));
        report["false_replacement_rate"] =
            json!(ratio(self.clips_with_false_replacements, self.clips));
        report
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

/// Ignore case and internal spacing, but never match a substring of a word.
/// Thus OpenCode == open code, while Hex != hexadecimal and Kit != toolkit.
fn term_count(text: &str, term: &str) -> usize {
    let target = normalized_words(term).concat();
    if target.is_empty() {
        return 0;
    }
    let words = normalized_words(text);
    let mut count = 0;
    let mut start = 0;
    while start < words.len() {
        let mut candidate = String::new();
        let mut next = start + 1;
        for (end, word) in words.iter().enumerate().skip(start) {
            candidate.push_str(word);
            if candidate == target {
                count += 1;
                next = end + 1;
                break;
            }
            if !target.starts_with(&candidate) {
                break;
            }
        }
        start = next;
    }
    count
}

pub fn run(manifest_path: &Path, warmups: usize, runs: usize, backend: Backend) -> Result<()> {
    if warmups == 0 {
        bail!("benchmark requires at least one warmup run");
    }
    if runs == 0 {
        bail!("benchmark requires at least one measured run");
    }
    let manifest_data = fs::read(manifest_path)
        .wrap_err_with(|| format!("could not read {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_data)
        .wrap_err_with(|| format!("could not parse {}", manifest_path.display()))?;
    validate_manifest(&manifest)?;
    let vocabulary = manifest.vocabulary;
    let model_id = backend.model_id();
    let trailing_silence = crate::transcription::trailing_silence_samples(model_id);
    let bias_record = backend.bias_record();
    if let Some(note) = backend.unrecognised_model_note() {
        eprintln!("{note}");
    }
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let clips = manifest
        .clips
        .into_iter()
        .map(|definition| load_clip(root, definition, model_id))
        .collect::<Result<Vec<_>>>()?;
    let total_audio_ms = clips.iter().map(|clip| clip.audio_ms).sum::<f64>();

    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let load_started = Instant::now();
    let mut backend = LoadedBackend::load(backend)?;
    let load_ms = elapsed_ms(load_started);
    let backend_name = backend.name().to_string();
    let mut model_record = json!({
        "type": "model",
        "backend": backend_name,
        "load_ms": load_ms,
        "clips": clips.len(),
        "audio_ms": total_audio_ms,
        "warmups": warmups,
        "runs": runs,
        "scoring_vocabulary": vocabulary,
        "trailing_silence_samples": trailing_silence,
    });
    if let Some(bias_record) = bias_record {
        // Only with the flag.
        model_record["bias_vocabulary"] = bias_record;
    }
    write_json(&mut output, model_record)?;

    let mut first_warmup_ms = None;
    let mut first_inference_ms = None;
    for warmup in 0..warmups {
        let mut inference_ms = 0.0;
        for clip in &clips {
            let inference_started = Instant::now();
            backend.transcribe(&clip.samples)?;
            let clip_inference_ms = elapsed_ms(inference_started);
            first_inference_ms.get_or_insert(clip_inference_ms);
            inference_ms += clip_inference_ms;
            write_json(
                &mut output,
                json!({
                    "type": "warmup_clip",
                    "backend": backend_name,
                    "warmup": warmup + 1,
                    "id": clip.id,
                    "inference_ms": clip_inference_ms,
                }),
            )?;
        }
        first_warmup_ms.get_or_insert(inference_ms);
        write_json(
            &mut output,
            json!({
                "type": "warmup",
                "backend": backend_name,
                "warmup": warmup + 1,
                "inference_ms": inference_ms,
                "audio_ms": total_audio_ms,
                "rtf": inference_ms / total_audio_ms,
            }),
        )?;
    }

    let mut run_times = Vec::with_capacity(runs);
    let mut clip_times = Vec::with_capacity(runs * clips.len());
    let mut total_word_errors = 0;
    let mut total_reference_words = 0;
    let mut quality = Quality::default();
    let mut groups = BTreeMap::<String, Quality>::new();
    for run in 0..runs {
        let mut inference_ms = 0.0;
        for clip in &clips {
            let inference_started = Instant::now();
            let transcript = backend.transcribe(&clip.samples)?;
            let clip_inference_ms = elapsed_ms(inference_started);
            inference_ms += clip_inference_ms;
            clip_times.push(clip_inference_ms);
            write_json(
                &mut output,
                json!({
                    "type": "measurement",
                    "backend": backend_name,
                    "run": run + 1,
                    "id": clip.id,
                    "inference_ms": clip_inference_ms,
                }),
            )?;
            if run == 0 {
                let mut clip_quality = Quality::default();
                clip_quality.record(clip, &transcript, &vocabulary);
                quality.record(clip, &transcript, &vocabulary);
                groups.entry(clip.group.clone()).or_default().record(
                    clip,
                    &transcript,
                    &vocabulary,
                );
                let (word_errors, reference_words) = word_errors(&clip.expected, &transcript);
                total_word_errors += word_errors;
                total_reference_words += reference_words;
                write_json(
                    &mut output,
                    json!({
                        "type": "clip",
                        "backend": backend_name,
                        "id": clip.id,
                        "audio_ms": clip.audio_ms,
                        "inference_ms": clip_inference_ms,
                        "rtf": clip_inference_ms / clip.audio_ms,
                        "word_errors": word_errors,
                        "reference_words": reference_words,
                        "expected": clip.expected,
                        "transcript": transcript,
                        "group": clip.group,
                        "quality": clip_quality.report(),
                    }),
                )?;
            }
        }
        run_times.push(inference_ms);
        write_json(
            &mut output,
            json!({
                "type": "run",
                "backend": backend_name,
                "run": run + 1,
                "inference_ms": inference_ms,
                "audio_ms": total_audio_ms,
                "rtf": inference_ms / total_audio_ms,
                "x_realtime": total_audio_ms / inference_ms,
            }),
        )?;
    }

    let median_run_ms = percentile(&run_times, 0.5);
    write_json(
        &mut output,
        json!({
            "type": "summary",
            "backend": backend_name,
            "load_ms": load_ms,
            "clips": clips.len(),
            "runs": runs,
            "audio_ms": total_audio_ms,
            "first_inference_ms": first_inference_ms,
            "first_warmup_ms": first_warmup_ms,
            "median_run_ms": median_run_ms,
            "median_rtf": median_run_ms / total_audio_ms,
            "median_x_realtime": total_audio_ms / median_run_ms,
            "median_clip_ms": percentile(&clip_times, 0.5),
            "p95_clip_ms": percentile(&clip_times, 0.95),
            "max_clip_ms": percentile(&clip_times, 1.0),
            "word_errors": total_word_errors,
            "reference_words": total_reference_words,
            "wer": total_word_errors as f64 / total_reference_words as f64,
            "quality": quality.report(),
            "groups": groups.iter().map(|(name, quality)| (name, quality.report())).collect::<BTreeMap<_, _>>(),
        }),
    )?;
    output.flush()?;
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.clips.is_empty() {
        bail!("benchmark manifest contains no clips");
    }
    let mut ids = HashSet::new();
    let vocabulary = manifest
        .vocabulary
        .iter()
        .map(|term| normalized_words(term).concat())
        .collect::<HashSet<_>>();
    if vocabulary.contains("") || vocabulary.len() != manifest.vocabulary.len() {
        bail!("scoring vocabulary must contain unique, nonempty terms");
    }
    for clip in &manifest.clips {
        if clip.id.trim().is_empty() {
            bail!("benchmark clip id cannot be empty");
        }
        if !ids.insert(&clip.id) {
            bail!("benchmark clip id is duplicated: {}", clip.id);
        }
        if normalized_words(&clip.expected).is_empty() {
            bail!("benchmark clip reference cannot be empty: {}", clip.id);
        }
        if !["target", "distractor", "ordinary", "private"].contains(&clip.group.as_str()) {
            bail!("unknown benchmark group: {}", clip.group);
        }
        if clip.group == "target" && clip.expected_terms.is_empty() {
            bail!("target clip has no expected terms: {}", clip.id);
        }
        if ["distractor", "ordinary"].contains(&clip.group.as_str())
            && !clip.expected_terms.is_empty()
        {
            bail!("guard clip cannot have expected terms: {}", clip.id);
        }
        let mut terms = HashSet::new();
        for term in &clip.expected_terms {
            let normalized = normalized_words(term).concat();
            if !vocabulary.contains(&normalized)
                || !terms.insert(normalized)
                || term_count(&clip.expected, term) == 0
            {
                bail!(
                    "expected term must be unique, in vocabulary and reference: {} ({term})",
                    clip.id
                );
            }
        }
    }
    Ok(())
}

fn load_clip(
    root: &Path,
    definition: ClipDefinition,
    model: Option<TranscriptionModelId>,
) -> Result<Clip> {
    let path = root.join(&definition.audio);
    let mut reader =
        WavReader::open(&path).wrap_err_with(|| format!("could not read {}", path.display()))?;
    let spec = reader.spec();
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 1..=8) => reader
            .samples::<i8>()
            .map(|sample| sample.map(|sample| f32::from(sample) / 128.0))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 9..=16) => reader
            .samples::<i16>()
            .map(|sample| {
                sample.map(|sample| f32::from(sample) / 2_f32.powi(spec.bits_per_sample as i32 - 1))
            })
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 17..=32) => reader
            .samples::<i32>()
            .map(|sample| {
                sample.map(|sample| sample as f32 / 2_f32.powi(spec.bits_per_sample as i32 - 1))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(eyre!(
                "unsupported WAV format in {}: {:?}, {} bits",
                path.display(),
                spec.sample_format,
                spec.bits_per_sample
            ));
        }
    };
    let channels = usize::from(spec.channels);
    if channels == 0 || samples.is_empty() || !samples.len().is_multiple_of(channels) {
        bail!("{} does not contain complete audio frames", path.display());
    }
    let mono = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect::<Vec<_>>();
    let audio_ms = mono.len() as f64 * 1_000.0 / f64::from(spec.sample_rate);
    let mut samples = resample_for_parakeet(&mono, spec.sample_rate);
    pad_gguf_samples(&mut samples, model);
    Ok(Clip {
        id: definition.id,
        expected: definition.expected,
        samples,
        audio_ms,
        expected_terms: definition.expected_terms,
        group: definition.group,
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn word_errors(expected: &str, actual: &str) -> (usize, usize) {
    let expected = normalized_words(expected);
    let actual = normalized_words(actual);
    let mut previous = (0..=actual.len()).collect::<Vec<_>>();
    for (expected_index, expected_word) in expected.iter().enumerate() {
        let mut current = vec![expected_index + 1];
        for (actual_index, actual_word) in actual.iter().enumerate() {
            current.push(if expected_word == actual_word {
                previous[actual_index]
            } else {
                1 + previous[actual_index]
                    .min(previous[actual_index + 1])
                    .min(current[actual_index])
            });
        }
        previous = current;
    }
    (previous[actual.len()], expected.len())
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn write_json(output: &mut impl Write, value: serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *output, &value)?;
    output.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use hound::{WavSpec, WavWriter};

    use super::*;

    #[test]
    fn word_error_rate_normalizes_case_and_punctuation() {
        assert_eq!(
            word_errors("Use OpenCode, please.", "use opencode please"),
            (0, 3)
        );
    }

    #[test]
    fn word_error_rate_counts_edits() {
        assert_eq!(word_errors("one two three", "one four"), (2, 3));
        assert_eq!(word_errors("one two", "zero one two three"), (2, 2));
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        assert_eq!(percentile(&[9.0, 1.0, 3.0, 2.0], 0.5), 2.0);
        assert_eq!(percentile(&[9.0, 1.0, 3.0, 2.0], 0.95), 9.0);
    }

    #[test]
    fn manifest_requires_unique_ids_and_spoken_references() {
        let manifest = |clips| Manifest {
            vocabulary: vec![],
            clips,
        };
        let clip = |id: &str, expected: &str| ClipDefinition {
            id: id.into(),
            audio: "unused.wav".into(),
            expected: expected.into(),
            expected_terms: vec![],
            group: ordinary_group(),
        };

        assert!(validate_manifest(&manifest(vec![])).is_err());
        assert!(validate_manifest(&manifest(vec![clip("blank", "...")])).is_err());
        assert!(
            validate_manifest(&manifest(vec![clip("same", "one"), clip("same", "two")])).is_err()
        );
        assert!(validate_manifest(&manifest(vec![clip("valid", "one")])).is_ok());
    }

    #[test]
    fn wav_preparation_downmixes_and_pads_without_changing_duration() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "hex-transcription-benchmark-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("stereo.wav");
        let mut writer = WavWriter::create(
            &path,
            WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in [1.0, -1.0, 0.5, 0.5] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        let clip = load_clip(
            &directory,
            ClipDefinition {
                id: "stereo".into(),
                audio: "stereo.wav".into(),
                expected: "reference".into(),
                expected_terms: vec![],
                group: ordinary_group(),
            },
            None,
        )
        .unwrap();

        assert_eq!(clip.audio_ms, 0.125);
        assert_eq!(clip.samples.len(), 24_000);
        assert_eq!(&clip.samples[..2], &[0.0, 0.5]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn vocabulary_scoring_respects_word_boundaries_and_counts_insertions() {
        assert_eq!(term_count("Use OPEN CODE and OpenCode.", "OpenCode"), 2);
        assert_eq!(term_count("hexadecimal toolkit effects runes", "Hex"), 0);
        assert_eq!(term_count("hexadecimal toolkit effects runes", "Kit"), 0);
        assert_eq!(term_count("a rune carved in stone", "Rune"), 1);
        let clip = Clip {
            id: "guard".into(),
            expected: "Tom Hanks found a rune".into(),
            expected_terms: vec![],
            group: "distractor".into(),
            samples: vec![],
            audio_ms: 1000.0,
        };
        let mut quality = Quality::default();
        quality.record(
            &clip,
            "Tom Hex found a rune",
            &["Hex".into(), "Rune".into()],
        );
        assert_eq!(quality.false_replacements, 1);
        assert_eq!(quality.clips_with_false_replacements, 1);
        assert_eq!(quality.report()["term_recall"], serde_json::Value::Null);
        assert_eq!(quality.word_errors, 1);
    }

    #[test]
    fn vocabulary_flag_parses_settings_shape() {
        let read = |json: &str| RunVocabulary::parse(json.as_bytes(), VocabularyBoost::default());
        let vocabulary = read(
            r#"["Hex", {"term": "  Qorvyx ", "sounds_like": ["corvix", "Corvix", "qorvyx"]}, "hex"]"#,
        )
        .unwrap();
        // Cleaned, deduplicated, and the same bias dictation builds.
        assert_eq!(vocabulary.input_terms, 2);
        assert_eq!(vocabulary.bias.0, ["Hex", "Qorvyx"]);
        assert_eq!(vocabulary.bias.1.sounds_like, [(1, "corvix".to_string())]);
        assert_eq!(vocabulary.bias.1.boost, Some(2.5));

        let long = format!(r#"["{}"]"#, "x".repeat(65));
        let too_many = format!(
            r#"[{{"term": "a", "sounds_like": {:?}}}]"#,
            (0..17).map(|n| format!("s{n}")).collect::<Vec<_>>()
        );
        for (input, wanted) in [
            (r#"{"term": "x"}"#, "at ."),
            (r#"[1]"#, "at [0]"),
            (r#"["ok", {"sounds_like": []}]"#, "at [1]"),
            (r#"["ok", {"term": "x", "sounds_like": null}]"#, "at [1]"),
            (r#"["ok", {"term": "x", "sounds_like": [2]}]"#, "at [1]"),
            (r#"["ok", "  "]"#, "at [1]"),
            (r#"[]"#, "no terms"),
            ("not json", "at ."),
            (long.as_str(), "longer than 64"),
            (too_many.as_str(), "at most 16"),
        ] {
            let error = read(input)
                .err()
                .unwrap_or_else(|| panic!("{input} was accepted"));
            assert!(error.contains(wanted), "{input}: {error}");
        }
    }

    #[test]
    fn vocabulary_files_that_cannot_be_read_name_the_path() {
        let missing = Path::new("/nonexistent/hex-vocabulary.json");
        let error = RunVocabulary::read(missing, VocabularyBoost::Normal)
            .err()
            .unwrap();
        assert!(
            error.contains("/nonexistent/hex-vocabulary.json"),
            "{error}"
        );
        let directory = std::env::temp_dir().join(format!("hex-vocab-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("bad.json");
        fs::write(&path, "[1,").unwrap();
        let error = RunVocabulary::read(&path, VocabularyBoost::Normal)
            .err()
            .unwrap();
        assert!(error.contains("bad.json"), "{error}");
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn the_record_reports_the_terms_the_engine_takes_not_the_terms_read() {
        let list = (0..70)
            .map(|n| format!(r#""term{n}""#))
            .collect::<Vec<_>>()
            .join(",");
        let vocabulary =
            RunVocabulary::parse(format!("[{list}]").as_bytes(), VocabularyBoost::Strong).unwrap();
        let record = vocabulary.record();
        assert_eq!(record["terms"], 64);
        assert_eq!(record["input_terms"], 70);
        assert_eq!(record["boost"], "strong");
    }

    #[test]
    fn vocabulary_is_refused_with_a_reason_per_model_family() {
        use TranscriptionModelId::*;
        assert_eq!(require_bias_model(None), Ok(()));
        assert_eq!(require_bias_model(Some(ParakeetUnifiedEnglish)), Ok(()));
        assert_eq!(require_bias_model(Some(ParakeetV3)), Ok(()));
        let whisper = require_bias_model(Some(WhisperLargeV3Turbo)).unwrap_err();
        assert!(whisper.contains("context prompt"), "{whisper}");
        let qwen = require_bias_model(Some(Qwen3Asr06B)).unwrap_err();
        assert!(qwen.contains("context prompt"), "{qwen}");
        let cohere = require_bias_model(Some(CohereTranscribe)).unwrap_err();
        assert!(cohere.contains("takes no vocabulary"), "{cohere}");
    }

    #[test]
    fn no_flag_no_bias() {
        use crate::transcription::models::{ModelRuntime, definition};
        let ModelRuntime::Gguf(artifact) =
            definition(TranscriptionModelId::ParakeetUnifiedEnglish).runtime
        else {
            panic!("Unified English is a GGUF model");
        };
        let directory = std::env::temp_dir().join(format!("hex-no-flag-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join(artifact.filename);
        let file = fs::File::create(&path).unwrap();
        file.set_len(artifact.bytes).unwrap();

        // Without the flag nothing is biased and the record says nothing of it.
        let backend = Backend::transcribe_cpp(path.clone(), None, None).unwrap();
        assert!(matches!(
            backend,
            Backend::TranscribeCpp {
                vocabulary: None,
                ..
            }
        ));
        assert!(backend.bias_record().is_none());
        // The catalog file is recognised by name and size, and padded as dictation pads.
        assert_eq!(
            backend.model_id(),
            Some(TranscriptionModelId::ParakeetUnifiedEnglish)
        );
        assert_eq!(
            crate::transcription::trailing_silence_samples(backend.model_id()),
            6_400
        );
        assert!(backend.unrecognised_model_note().is_none());

        // The same name at another size is no catalog artifact.
        file.set_len(artifact.bytes - 1).unwrap();
        let backend = Backend::transcribe_cpp(path, None, None).unwrap();
        assert_eq!(backend.model_id(), None);
        assert!(backend.unrecognised_model_note().is_some());
        assert!(Backend::Onnx.bias_record().is_none());
        assert_eq!(Backend::Onnx.model_id(), None);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn an_unnamed_file_takes_the_bias_check_to_load_and_a_bad_vocabulary_names_its_path() {
        let directory =
            std::env::temp_dir().join(format!("hex-vocab-backend-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let vocabulary = directory.join("vocabulary.json");
        fs::write(&vocabulary, r#"["Hex"]"#).unwrap();

        // A file that is no catalog artifact is checked once it loads.
        let unknown = Backend::transcribe_cpp(
            PathBuf::from("renamed.gguf"),
            Some(&vocabulary),
            Some(VocabularyBoost::Light),
        )
        .unwrap();
        assert_eq!(unknown.model_id(), None);
        assert_eq!(unknown.bias_record().unwrap()["boost"], "light");
        // A path that does not exist is no "not a catalog artifact" note.
        assert!(unknown.unrecognised_model_note().is_none());

        let error = Backend::transcribe_cpp(
            PathBuf::from("renamed.gguf"),
            Some(&directory.join("missing.json")),
            None,
        )
        .err()
        .unwrap();
        assert!(error.starts_with("--vocabulary could not read"), "{error}");
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn clips_pad_like_dictation_for_unified_english_only() {
        let directory = std::env::temp_dir().join(format!("hex-pad-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("one.wav");
        let mut writer = WavWriter::create(
            &path,
            WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        for _ in 0..32_000 {
            writer.write_sample(0.1f32).unwrap();
        }
        writer.finalize().unwrap();
        let load = |model| {
            load_clip(
                &directory,
                ClipDefinition {
                    id: "one".into(),
                    audio: "one.wav".into(),
                    expected: "x".into(),
                    expected_terms: vec![],
                    group: ordinary_group(),
                },
                model,
            )
            .unwrap()
            .samples
            .len()
        };
        let plain = load(None);
        assert_eq!(
            load(Some(TranscriptionModelId::ParakeetUnifiedEnglish)),
            plain + 6_400
        );
        assert_eq!(load(Some(TranscriptionModelId::ParakeetV3)), plain);
        fs::remove_dir_all(directory).unwrap();
    }
}
