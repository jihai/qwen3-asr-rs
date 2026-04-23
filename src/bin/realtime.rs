//! Real-time speech-to-text from microphone.
//!
//! Usage:
//!   cargo run --bin realtime --features realtime --release
//!
//! Environment variables (or .env file):
//!   QWEN3_ASR_MODEL_DIR  — path to model directory (required if --model-dir not set)
//!
//! Examples:
//!   cargo run --bin realtime --features realtime --release
//!   cargo run --bin realtime --features realtime --release -- --language english
//!   cargo run --bin realtime --features realtime --release -- --chunk-size 1.0
//!   cargo run --bin realtime --features realtime --release -- --no-save

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

#[derive(Parser)]
#[command(name = "realtime", about = "Real-time speech-to-text from microphone")]
struct Args {
    /// Path to model directory. Falls back to QWEN3_ASR_MODEL_DIR env var.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Force a specific language (e.g. "english", "chinese"). Auto-detects if omitted.
    #[arg(long)]
    language: Option<String>,

    /// Audio chunk size in seconds (smaller = lower latency, larger = more accurate).
    #[arg(long, default_value = "2.0")]
    chunk_size: f32,

    /// Number of initial cold-start chunks before prefix conditioning kicks in.
    #[arg(long, default_value = "2")]
    unfixed_chunk_num: usize,

    /// Number of tokens to roll back from the end for correction. Higher = more self-correction.
    #[arg(long, default_value = "5")]
    unfixed_token_num: usize,

    /// Maximum new tokens per streaming step.
    #[arg(long, default_value = "32")]
    max_tokens_streaming: usize,

    /// Maximum new tokens for the final flush.
    #[arg(long, default_value = "512")]
    max_tokens_final: usize,

    /// Directory for transcript files. Default: transcripts/
    #[arg(long, default_value = "transcripts")]
    transcript_dir: PathBuf,

    /// Disable automatic transcript saving.
    #[arg(long)]
    no_save: bool,
}

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    let args = Args::parse();

    // Resolve model directory
    let model_dir = args
        .model_dir
        .or_else(|| std::env::var("QWEN3_ASR_MODEL_DIR").ok().map(PathBuf::from))
        .expect("Provide --model-dir or set QWEN3_ASR_MODEL_DIR");

    // Print configuration
    eprintln!("=== Configuration ===");
    eprintln!("  model_dir           : {}", model_dir.display());
    eprintln!("  language            : {}", args.language.as_deref().unwrap_or("auto"));
    eprintln!("  chunk_size          : {}s", args.chunk_size);
    eprintln!("  unfixed_chunk_num   : {}", args.unfixed_chunk_num);
    eprintln!("  unfixed_token_num   : {}", args.unfixed_token_num);
    eprintln!("  max_tokens_streaming: {}", args.max_tokens_streaming);
    eprintln!("  max_tokens_final    : {}", args.max_tokens_final);
    eprintln!("  save transcripts    : {}", if args.no_save { "no" } else { "yes" });
    eprintln!();

    eprintln!("Loading model from {}...", model_dir.display());
    let device = qwen3_asr::best_device();
    let engine = qwen3_asr::AsrInference::load(&model_dir, device)?;
    eprintln!("Model loaded.");

    // Set up streaming options
    let mut opts = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(args.chunk_size)
        .with_unfixed_chunk_num(args.unfixed_chunk_num)
        .with_unfixed_token_num(args.unfixed_token_num)
        .with_max_new_tokens_streaming(args.max_tokens_streaming)
        .with_max_new_tokens_final(args.max_tokens_final);
    if let Some(lang) = &args.language {
        opts = opts.with_language(lang.as_str());
    }
    let mut state = engine.init_streaming(opts);

    // Set up transcript file
    let mut output_file = if !args.no_save {
        let path = create_transcript_path(&args.transcript_dir)?;
        eprintln!("Transcript: {}", path.display());
        Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?,
        )
    } else {
        None
    };

    // Set up mic capture
    let host = cpal::default_host();
    let mic = host
        .default_input_device()
        .expect("no input device found");
    eprintln!("Mic: {}", mic.name()?);

    let mic_config = mic.default_input_config()?;
    let sample_rate = mic_config.sample_rate().0;
    let channels = mic_config.channels() as usize;
    eprintln!("Mic config: {}Hz, {} ch", sample_rate, channels);

    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let needs_resample = sample_rate != 16000;

    let stream = mic.build_input_stream(
        &mic_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = if channels == 1 {
                data.to_vec()
            } else {
                data.chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect()
            };
            let _ = tx.send(mono);
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;
    stream.play()?;

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::Relaxed);
    })?;

    eprintln!("\nListening... (Ctrl+C to stop)\n");

    // Output state: track committed (permanent) text vs active (evolving) tail
    let mut committed = String::new();
    let mut last_text = String::new();

    // Resampler state
    let resample_block = 1024usize;
    let mut resample_buf: Vec<f32> = Vec::new();
    let mut resampler = if needs_resample {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = 16000.0 / sample_rate as f64;
        Some(SincFixedIn::<f32>::new(ratio, 2.0, params, resample_block, 1).unwrap())
    } else {
        None
    };

    while running.load(Ordering::Relaxed) {
        let chunk = match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(c) => c,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        let samples_16k = if let Some(ref mut rs) = resampler {
            resample_buf.extend_from_slice(&chunk);
            let mut out = Vec::new();
            while resample_buf.len() >= resample_block {
                let block: Vec<f32> = resample_buf.drain(..resample_block).collect();
                if let Ok(resampled) = rs.process(&[block], None) {
                    if let Some(ch) = resampled.into_iter().next() {
                        out.extend(ch);
                    }
                }
            }
            if out.is_empty() {
                continue;
            }
            out
        } else {
            chunk
        };

        match engine.feed_audio(&mut state, &samples_16k) {
            Ok(Some(result)) => {
                if result.text != last_text {
                    display_streaming(&result.text, &mut committed, &mut output_file);
                    last_text = result.text;
                }
            }
            Ok(None) => {}
            Err(e) => eprintln!("\nInference error: {}", e),
        }
    }

    // Finalize
    eprint!("\r\x1b[2K");
    eprintln!("Finalizing...");
    let final_result = engine.finish_streaming(&mut state)?;

    // Commit any remaining text
    let final_text = final_result.text.trim();
    if !final_text.is_empty() {
        let committed_trimmed = committed.trim_end();
        let new_part = if !committed_trimmed.is_empty() && final_text.starts_with(committed_trimmed) {
            final_text[committed_trimmed.len()..].trim()
        } else {
            final_text
        };
        if !new_part.is_empty() {
            eprintln!("{}", new_part);
            if let Some(f) = &mut output_file {
                let _ = writeln!(f, "{}", new_part);
                let _ = f.flush();
            }
        }
    }

    if !final_result.language.is_empty() {
        eprintln!("\n[Language: {}]", final_result.language);
    }

    if let Some(f) = &mut output_file {
        let _ = f.flush();
        eprintln!("[Transcript saved]");
    }

    Ok(())
}

/// Display streaming output with sentence-based line commits.
///
/// The streaming engine can **revise** earlier text via rollback, so we cannot
/// assume the new text always extends the committed prefix. We validate with
/// `starts_with` and fall back to the longest common prefix on mismatch.
fn display_streaming(
    full_text: &str,
    committed: &mut String,
    output_file: &mut Option<std::fs::File>,
) {
    let text = full_text.trim();
    if text.is_empty() {
        return;
    }

    // Step 1: ensure committed is still a valid prefix of the new text.
    // The model's rollback can revise earlier tokens, invalidating committed.
    let committed_trimmed = committed.trim_end();
    if !committed_trimmed.is_empty() && !text.starts_with(committed_trimmed) {
        *committed = longest_common_prefix(committed_trimmed, text);
    }

    // Step 2: look for sentence boundaries in the portion after committed.
    let remaining = &text[committed.len()..];
    if let Some(end_idx) = find_sentence_end(remaining) {
        let to_commit = &remaining[..end_idx];
        let to_commit_trimmed = to_commit.trim();
        if !to_commit_trimmed.is_empty() {
            eprint!("\r\x1b[2K");
            eprintln!("{}", to_commit_trimmed);

            if let Some(f) = output_file {
                let _ = writeln!(f, "{}", to_commit_trimmed);
                let _ = f.flush();
            }
        }
        // Advance committed to include the newly committed portion.
        let new_len = committed.len() + end_idx;
        *committed = text[..new_len].to_string();
    }

    // Step 3: show the evolving tail on the active (overwritten) line.
    let tail = text[committed.len()..].trim();
    eprint!("\r\x1b[2K{}", tail);
    std::io::stderr().flush().ok();
}

/// Find the byte index *past* the last sentence-ending character in `text`.
/// Returns an exclusive end index suitable for `&text[..idx]`.
fn find_sentence_end(text: &str) -> Option<usize> {
    let mut last = None;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?' | '\u{3002}' | '\u{FF01}' | '\u{FF1F}' | '\u{2026}') {
            last = Some(i + c.len_utf8());
        }
    }
    last
}

/// Return the longest common prefix of two strings (always at a char boundary).
fn longest_common_prefix(a: &str, b: &str) -> String {
    let end = a
        .char_indices()
        .zip(b.chars())
        .take_while(|((_, ca), cb)| ca == cb)
        .last()
        .map(|((i, c), _)| i + c.len_utf8())
        .unwrap_or(0);
    a[..end].to_string()
}

/// Create a transcript file path: `{dir}/YYYY-MM-DD-{id}.txt`
/// where id auto-increments starting from 1 for each day.
fn create_transcript_path(dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let date = local_date_string();

    // Find next available ID for today
    let mut id = 1u32;
    loop {
        let name = format!("{}-{}.txt", date, id);
        let path = dir.join(&name);
        if !path.exists() {
            return Ok(path);
        }
        id += 1;
    }
}

/// Get local date as YYYY-MM-DD string.
fn local_date_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as libc::time_t;
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&secs, &mut tm);
        format!(
            "{:04}-{:02}-{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday
        )
    }
}

