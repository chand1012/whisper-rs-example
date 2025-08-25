use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
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
    file: Option<PathBuf>,

    /// Path to save the transcription output (defaults to <input_file>.txt)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Enable real-time transcription from microphone
    #[arg(long, short = 'm')]
    mic: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(&args.model.to_string_lossy(), ctx_params)
        .context("Failed to load model")?;

    if args.mic {
        run_mic_mode(&ctx)?;
        return Ok(());
    }

    let input_path = args
        .file
        .as_ref()
        .context("--file is required when not using -m/--mic")?;

    // Create parameters for transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(true);
    params.set_print_timestamps(true);
    params.set_tdrz_enable(true);

    // Load and decode the audio file
    let audio_data = load_audio(input_path)?;

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
    let output_path = args.output.clone().unwrap_or_else(|| {
        let mut path = input_path.clone();
        let mut out = PathBuf::from(path);
        out.set_extension("txt");
        out
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

fn run_mic_mode(ctx: &WhisperContext) -> Result<()> {
    // Configuration similar to whisper.cpp stream example
    let step_ms: i32 = 3000; // inference every 3s
    let length_ms: i32 = 10000; // analyze up to 10s window
    let keep_ms: i32 = 200; // keep 200ms overlap

    let sample_rate_target: u32 = 16000;

    // Shared buffer for captured audio (mono f32 at device sample rate)
    let captured: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(
        (sample_rate_target as usize) * 30,
    )));

    // Setup CPAL input stream
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No default input device")?;
    let mut supported_configs = device
        .supported_input_configs()
        .context("No supported input configs")?;
    // pick a reasonable config (prefer 16k mono f32 if available)
    let mut chosen_config = None;
    while let Some(range) = supported_configs.next() {
        if range.sample_format() == cpal::SampleFormat::F32 {
            let cfg = range.with_max_sample_rate();
            chosen_config = Some(cfg);
            break;
        }
    }
    let config = chosen_config
        .map(|c| c.config())
        .unwrap_or_else(|| cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(48000),
            buffer_size: cpal::BufferSize::Default,
        });

    let input_sample_rate = config.sample_rate.0;

    let captured_clone = captured.clone();
    let err_fn = |err| eprintln!("Input stream error: {err}");

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            let mut buf = captured_clone.lock().unwrap();
            if config.channels == 1 {
                buf.extend_from_slice(data);
            } else {
                // downmix to mono by taking channel 0
                let mut i = 0usize;
                while i < data.len() {
                    buf.push(data[i]);
                    i += config.channels as usize;
                }
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    println!("[Start speaking] (Ctrl+C to stop)");

    // Whisper state
    let mut state = ctx.create_state().context("Failed to create state")?;

    // rolling buffers
    let mut previous_window: Vec<f32> = Vec::new();
    let mut last_infer = Instant::now();
    let step = Duration::from_millis(step_ms as u64);
    let max_len_samples = (length_ms as u32 as usize) * (sample_rate_target as usize) / 1000;
    let keep_len_samples = (keep_ms as u32 as usize) * (sample_rate_target as usize) / 1000;

    loop {
        // sleep a bit to avoid busy loop
        std::thread::sleep(Duration::from_millis(10));

        if last_infer.elapsed() < step {
            continue;
        }
        last_infer = Instant::now();

        // pull data
        let mut data = {
            let mut buf = captured.lock().unwrap();
            let mut out = Vec::new();
            std::mem::swap(&mut *buf, &mut out);
            out
        };

        if data.is_empty() {
            continue;
        }

        // resample to 16k if needed (simple linear)
        let resampled = if input_sample_rate != sample_rate_target {
            simple_resample_linear(&data, input_sample_rate, sample_rate_target)
        } else {
            data
        };

        // compose current window: keep tail of previous + new
        let mut window = Vec::new();
        if !previous_window.is_empty() {
            let take = previous_window.len().min(keep_len_samples);
            window.extend_from_slice(&previous_window[previous_window.len() - take..]);
        }
        window.extend_from_slice(&resampled);

        // clamp to max length
        if window.len() > max_len_samples {
            window = window[window.len() - max_len_samples..].to_vec();
        }

        // set params each step
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_timestamps(true);
        params.set_single_segment(false);
        params.set_n_threads(num_cpus::get().min(4) as i32);

        if state.full(params, &window).is_ok() {
            if let Ok(n_segments) = state.full_n_segments() {
                for i in 0..n_segments {
                    if let (Ok(t0), Ok(t1), Ok(text)) = (
                        state.full_get_segment_t0(i),
                        state.full_get_segment_t1(i),
                        state.full_get_segment_text(i),
                    ) {
                        println!(
                            "[{} --> {}] {}",
                            format_ts_ms(t0 as i64),
                            format_ts_ms(t1 as i64),
                            text
                        );
                    }
                }
                println!();
            }
        }

        // remember tail for next iteration
        previous_window = window;
    }
}

fn simple_resample_linear(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }
    let scale = to_hz as f32 / from_hz as f32;
    let new_len = (input.len() as f32 * scale) as usize;
    let mut resampled = Vec::with_capacity(new_len);
    let last = input.len().saturating_sub(1);
    for i in 0..new_len {
        let pos = i as f32 / scale;
        let pf = pos.floor();
        let pc = pos.ceil();
        let i0 = pf as usize;
        let i1 = pc as usize;
        if i1 > last {
            resampled.push(input[last]);
            continue;
        }
        let t = pos - pf;
        let s = input[i0] * (1.0 - t) + input[i1] * t;
        resampled.push(s);
    }
    resampled
}

fn format_ts_ms(ms: i64) -> String {
    let seconds = ms as f32 / 100.0; // incoming t0/t1 are in 10ms units in whisper.cpp; here we output raw
    format!("{:.2}", seconds)
}
