use anyhow::{Context, Result};
use clap::Parser;
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};
use symphonia::core::{
    audio::{AudioBuffer, Signal},
    codecs::DecoderOptions,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Transcribe audio and video files using whisper-rs"
)]
struct Args {
    /// Path to the whisper model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the input audio/video file
    #[arg(short, long)]
    file: PathBuf,

    /// Path to save the transcription output (defaults to <input_file>.txt)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let params = WhisperContextParameters::default();
    // Load the model
    let ctx = WhisperContext::new_with_params(&args.model.to_string_lossy(), params)
        .context("Failed to load model")?;

    // Create parameters for transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(true);
    params.set_print_timestamps(true);

    // Load and decode the audio file
    let audio_data = load_audio(&args.file)?;

    // Create a state for transcription
    let mut state = ctx.create_state().context("Failed to create state")?;

    // Run the model
    state
        .full(params, &audio_data)
        .context("Failed to run model")?;

    // Get the number of segments
    let num_segments = state
        .full_n_segments()
        .context("Failed to get number of segments")?;

    // Determine output path
    let output_path = args.output.unwrap_or_else(|| {
        let mut path = args.file.clone();
        path.set_extension("txt");
        path
    });

    // Write the transcription to file
    let mut output_file = File::create(&output_path).context("Failed to create output file")?;

    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .context("Failed to get segment text")?;
        let start = state
            .full_get_segment_t0(i)
            .context("Failed to get segment start time")?;
        let end = state
            .full_get_segment_t1(i)
            .context("Failed to get segment end time")?;

        writeln!(output_file, "[{:.2} - {:.2}] {}", start, end, segment)
            .context("Failed to write to output file")?;
    }

    println!("Transcription saved to: {}", output_path.display());
    Ok(())
}

fn load_audio(path: &Path) -> Result<Vec<f32>> {
    // Create a media source from the file
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a hint to help with format detection
    let hint = Hint::new();

    // Use default format options
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    // Use default decoder options
    let decoder_opts = DecoderOptions::default();

    // Probe the media source to determine the format
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe media format")?;

    // Get the format reader and find the first audio track
    let mut format = probed.format;
    let tracks = format.tracks();
    let track = tracks
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found")?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);

    // Create a decoder for the track
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    let mut sample_buf = Vec::new();

    // Decode the audio
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;

        // Convert to f32 samples
        let mut buf = AudioBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        decoded.convert(&mut buf);

        // Only take the first channel if there are multiple
        sample_buf.extend_from_slice(buf.chan(0));
    }

    // Resample to 16KHz if needed
    let sample_rate = sample_rate as f32;
    if sample_rate != 16000.0 {
        // Simple linear interpolation resampling
        let scale = 16000.0 / sample_rate;
        let new_len = (sample_buf.len() as f32 * scale) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let pos = i as f32 / scale;
            let pos_floor = pos.floor() as usize;
            let pos_ceil = pos.ceil() as usize;

            if pos_ceil >= sample_buf.len() {
                break;
            }

            let t = pos - pos_floor as f32;
            let sample = sample_buf[pos_floor] * (1.0 - t) + sample_buf[pos_ceil] * t;
            resampled.push(sample);
        }

        Ok(resampled)
    } else {
        Ok(sample_buf)
    }
}
