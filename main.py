import argparse
import importlib.util
import whisperx
import re
import gc
import os
import torch
from qwen_asr import Qwen3ASRModel
from typing import List, Dict

_original_torch_load = torch.load

def _trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _trusted_load

QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_SAMPLE_RATE = 16000
QWEN_CHUNK_SECONDS = 60
DEFAULT_BATCH_SIZE = 4
QWEN_MAX_NEW_TOKENS = 2048
QWEN_TO_WHISPER_LANGUAGE = {
    "chinese": "zh",
    "english": "en",
    "cantonese": "zh",
    "arabic": "ar",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "indonesian": "id",
    "italian": "it",
    "korean": "ko",
    "russian": "ru",
    "thai": "th",
    "vietnamese": "vi",
    "japanese": "ja",
    "turkish": "tr",
    "hindi": "hi",
    "malay": "ms",
    "dutch": "nl",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "polish": "pl",
    "czech": "cs",
    "filipino": "tl",
    "persian": "fa",
    "greek": "el",
    "romanian": "ro",
    "hungarian": "hu",
    "macedonian": "mk",
}


def _qwen_language_to_whisper_code(language: str) -> str | None:
    """Convert Qwen's detected language name to WhisperX's ISO code."""
    if not language:
        return None
    parts = [part.strip().lower() for part in re.split(r"[,/]", language) if part.strip()]
    if len(parts) != 1:
        return None
    return QWEN_TO_WHISPER_LANGUAGE.get(parts[0])


def _load_qwen_model(
    device: str,
    compute_type: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    attention_implementation: str = "auto",
) -> Qwen3ASRModel:
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    dtype = torch.float32 if device == "cpu" or compute_type == "float32" else torch.bfloat16
    device_map = "cuda:0" if device == "cuda" else "cpu"
    flash_attention_available = (
        device == "cuda" and importlib.util.find_spec("flash_attn") is not None
    )
    if attention_implementation == "auto":
        selected_attention = "flash_attention_2" if flash_attention_available else None
        if selected_attention is None:
            print(
                "FlashAttention 2 is unavailable for this run; using the Transformers default. "
                "Install it with: pip install -U flash-attn --no-build-isolation"
            )
    elif attention_implementation == "flash_attention_2":
        if not flash_attention_available:
            raise RuntimeError(
                "FlashAttention 2 was requested but flash-attn is not installed or CUDA is not in use. "
                "Install it with: pip install -U flash-attn --no-build-isolation"
            )
        selected_attention = "flash_attention_2"
    elif attention_implementation == "sdpa":
        selected_attention = "sdpa"
    else:
        raise ValueError(f"Unsupported attention implementation: {attention_implementation}")

    model_kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "forced_aligner": QWEN_FORCED_ALIGNER_MODEL,
        "forced_aligner_kwargs": {"dtype": dtype, "device_map": device_map},
        "max_inference_batch_size": batch_size,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
    }
    if selected_attention:
        model_kwargs["attn_implementation"] = selected_attention
        model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = selected_attention

    print(
        f"Loading Qwen ASR model '{QWEN_ASR_MODEL}' on {device} with dtype={dtype} "
        f"and forced aligner '{QWEN_FORCED_ALIGNER_MODEL}' "
        f"(attention={selected_attention or 'Transformers default'}, batch_size={batch_size})..."
    )
    try:
        return Qwen3ASRModel.from_pretrained(QWEN_ASR_MODEL, **model_kwargs)
    except Exception as exc:
        flash_attention_error = any(
            marker in str(exc).lower()
            for marker in ("flash_attn", "flash attention", "flash_attention")
        )
        if (
            attention_implementation != "auto"
            or selected_attention != "flash_attention_2"
            or not flash_attention_error
        ):
            raise

        print(
            f"FlashAttention 2 could not be initialized ({exc}); "
            "retrying with the Transformers default."
        )
        model_kwargs.pop("attn_implementation", None)
        model_kwargs["forced_aligner_kwargs"].pop("attn_implementation", None)
        if device == "cuda":
            torch.cuda.empty_cache()
        return Qwen3ASRModel.from_pretrained(QWEN_ASR_MODEL, **model_kwargs)


def _normalize_aligned_token(token: str) -> str:
    return "".join(
        character
        for character in token
        if character.isalnum() or character in "'’"
    ).casefold()


def _restore_transcript_punctuation(words: List[Dict], text: str) -> None:
    """Put Qwen's punctuation back onto the forced-aligner word spans."""
    transcript_tokens = text.split()
    token_index = 0

    for word_info in words:
        aligned_token = _normalize_aligned_token(word_info["word"])
        if not aligned_token:
            continue

        while token_index < len(transcript_tokens):
            transcript_token = transcript_tokens[token_index]
            token_index += 1
            normalized_token = _normalize_aligned_token(transcript_token)
            if not normalized_token:
                continue

            if normalized_token == aligned_token:
                word_info["word"] = transcript_token
                break


def _transcribe_qwen_chunks(
    model: Qwen3ASRModel,
    audio,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[List[Dict], List[str]]:
    """Transcribe one-minute chunks in batches with automatic language detection."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    chunk_samples = QWEN_CHUNK_SECONDS * QWEN_SAMPLE_RATE
    chunks = []
    segments = []
    detected_languages = []

    for chunk_start in range(0, len(audio), chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, len(audio))
        chunks.append({
            "start": chunk_start / QWEN_SAMPLE_RATE,
            "end": chunk_end / QWEN_SAMPLE_RATE,
            "audio": audio[chunk_start:chunk_end],
        })

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        first_chunk = batch_start + 1
        last_chunk = batch_start + len(batch)
        print(
            f"Transcribing Qwen chunks {first_chunk}–{last_chunk}/{len(chunks)} "
            f"({format_time_srt(batch[0]['start'])}–{format_time_srt(batch[-1]['end'])})..."
        )

        transcriptions = model.transcribe(
            [(chunk["audio"], QWEN_SAMPLE_RATE) for chunk in batch],
            return_time_stamps=True,
        )
        if len(transcriptions) != len(batch):
            raise RuntimeError(
                f"Qwen returned {len(transcriptions)} results for a batch of {len(batch)} chunks"
            )

        for chunk, transcription_result in zip(batch, transcriptions):
            start_seconds = chunk["start"]
            end_seconds = chunk["end"]
            detected_language = (transcription_result.language or "").strip()
            text = (transcription_result.text or "").strip()
            if detected_language:
                detected_languages.append(detected_language)
            if text:
                words = []
                previous_end = None
                for item in transcription_result.time_stamps or []:
                    word = (getattr(item, "text", "") or "").strip()
                    word_start = float(getattr(item, "start_time", 0.0))
                    word_end = float(getattr(item, "end_time", 0.0))
                    if previous_end is not None and word_end <= previous_end:
                        continue
                    if previous_end is not None:
                        word_start = max(word_start, previous_end)
                    if word and word_end > word_start:
                        words.append({
                            "word": word,
                            "start": start_seconds + word_start,
                            "end": start_seconds + word_end,
                        })
                        previous_end = word_end

                if words:
                    _restore_transcript_punctuation(words, text)
                    segments.append({
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "text": text,
                        "words": words,
                    })
                else:
                    segments.append({
                        "start": start_seconds,
                        "end": end_seconds,
                        "text": text,
                    })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return segments, list(dict.fromkeys(detected_languages))


def split_by_pauses(segment: Dict, min_pause: float = 0.3, max_duration: float = 5.0, max_chars: int = 80) -> List[Dict]:
    """Split speech at natural pauses while preserving readable chunks."""
    words = segment.get("words", [])
    if not words:
        return [segment]

    sub_segments = []
    current_phrase = []
    phrase_start = None
    phrase_end = None
    current_length = 0
    prev_word_weak_punct = False

    def flush_phrase() -> None:
        nonlocal current_phrase, phrase_start, phrase_end, current_length, prev_word_weak_punct
        if not current_phrase or phrase_start is None or phrase_end is None:
            return
        sub_segments.append({
            "start": phrase_start,
            "end": phrase_end,
            "text": " ".join([w.get("word", "").strip() for w in current_phrase]),
            "words": current_phrase,
        })
        current_phrase = []
        phrase_start = None
        phrase_end = None
        current_length = 0
        prev_word_weak_punct = False

    for word_info in words:
        word = word_info.get("word", "").strip()
        start = word_info.get("start")
        end = word_info.get("end")
        if not word or start is None or end is None:
            continue

        if phrase_start is None:
            phrase_start = start

        pause_duration = start - phrase_end if phrase_end is not None else 0.0
        pause_threshold = min_pause * 0.7 if prev_word_weak_punct else min_pause
        next_len = current_length + len(word) + (1 if current_phrase else 0)
        next_duration = end - phrase_start

        # Split BEFORE this word when a clear pause happened, or adding this word
        # would make the subtitle too long.
        if current_phrase and (
            pause_duration >= pause_threshold
            or next_len > max_chars
            or next_duration > max_duration
        ):
            flush_phrase()
            phrase_start = start

        if current_phrase:
            current_length += 1
        current_phrase.append(word_info)
        current_length += len(word)
        phrase_end = end
        prev_word_weak_punct = word.endswith((",", ";", ":"))

        # Split AFTER punctuation so punctuation stays with the sentence.
        if word.endswith((".", "!", "?")):
            flush_phrase()

    flush_phrase()
    return sub_segments if sub_segments else [segment]


def extend_subtitle_timing(sub_segments: List[Dict], linger_time: float = 0.5, safety_gap: float = 0.02) -> List[Dict]:
    """Extend subtitle end times into silent gaps without overlapping next subtitle."""
    if len(sub_segments) < 2:
        return sub_segments

    for i in range(len(sub_segments) - 1):
        current_end = float(sub_segments[i]["end"])
        next_start = float(sub_segments[i + 1]["start"])
        gap = next_start - current_end

        if gap <= safety_gap:
            continue

        extension = min(linger_time, gap - safety_gap)
        if extension > 0:
            sub_segments[i]["end"] = current_end + extension

    return sub_segments


def format_time_srt(seconds: float) -> str:
    """Format time in SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_video(
    video_path: str,
    output_dir: str | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = DEFAULT_BATCH_SIZE,
    attention_implementation: str = "auto",
):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    if output_dir is None:
        # Save SRT file in the same directory as the video file
        output_dir = os.path.dirname(video_path)
    else:
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_srt_path = os.path.join(output_dir, f"{base_name}.srt")

    print(f"Loading audio from: {video_path}")
    audio = whisperx.load_audio(video_path)
    duration_seconds = len(audio) / QWEN_SAMPLE_RATE

    model = _load_qwen_model(
        device,
        compute_type,
        batch_size=batch_size,
        attention_implementation=attention_implementation,
    )

    print("Transcribing audio...")
    segments, detected_languages = _transcribe_qwen_chunks(
        model,
        audio,
        batch_size=batch_size,
    )
    detected_language = ",".join(detected_languages)
    language_code = None
    if len(detected_languages) == 1:
        language_code = _qwen_language_to_whisper_code(detected_languages[0])

    print(f"Detected language(s): {detected_language or 'unknown'}")

    if not segments:
        print("No speech detected.")
        return

    result = {
        "language": language_code,
        "segments": segments,
    }

    has_word_timing = any(segment.get("words") for segment in result["segments"])
    if has_word_timing:
        print("Using Qwen forced-alignment word timestamps.")
    elif result["segments"] and language_code:
        print(f"Loading alignment model for {language_code}...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code, device=device
            )

            print("Aligning transcription with precise timestamps...")
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception as exc:
            print(f"Alignment unavailable; writing chunk-timed subtitles: {exc}")
    elif result["segments"]:
        print(
            f"No WhisperX alignment model mapping for '{detected_language}'. "
            "Writing chunk-timed subtitles."
        )

    print(f"Generating SRT file: {output_srt_path}")
    subtitle_counter = 1
    all_sub_segments = []

    for segment in result["segments"]:
        all_sub_segments.extend(split_by_pauses(segment, min_pause=0.3, max_duration=5.0, max_chars=80))

    all_sub_segments = extend_subtitle_timing(all_sub_segments, linger_time=0.5, safety_gap=0.02)
    
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for sub_segment in all_sub_segments:
            start_time = sub_segment["start"]
            end_time = sub_segment["end"]
            text = sub_segment["text"].strip()
            
            # Skip empty segments
            if not text:
                continue

            # Write subtitle index
            f.write(f"{subtitle_counter}\n")
            subtitle_counter += 1
            
            # Write start and end timestamps
            f.write(f"{format_time_srt(start_time)} --> {format_time_srt(end_time)}\n")
            
            # Write text (single line, no wrapping)
            f.write(f"{text}\n\n")

    print(f"Subtitles generated successfully and saved to: {output_srt_path}")

    del model
    if "model_a" in locals():
        del model_a
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

def process_videos_in_folder(
    folder_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = DEFAULT_BATCH_SIZE,
    attention_implementation: str = "auto",
):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return
    
    # Find all video files recursively
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in '{folder_path}' and its subdirectories.")
        return
    
    print(f"Found {len(video_files)} video file(s) to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    
    # Process each video file
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
        print(f"{'='*60}")
        
        # Check if SRT file already exists
        base_name = os.path.splitext(video_file)[0]
        srt_path = f"{base_name}.srt"
        
        if os.path.exists(srt_path):
            print(f"SRT file already exists: {srt_path}")
            user_input = input("Do you want to overwrite it? (y/n): ").lower().strip()
            if user_input != 'y':
                print("Skipping this video...")
                continue
        
        try:
            # Generate subtitles (output_dir=None means save in same directory as video)
            generate_srt_from_video(
                video_file,
                output_dir=None,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                attention_implementation=attention_implementation,
            )
            print(f"✓ Successfully processed: {video_file}")
        except Exception as e:
            print(f"✗ Error processing {video_file}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subtitles for video files in a folder or a specific video file.")
    parser.add_argument("input_path", type=str, help="Path to a video file or a folder containing video files.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for processing (cuda or cpu).")
    parser.add_argument("--compute_type", type=str, default="float16", choices=["float16", "int8", "float32"], help="Compute type to use (float16, int8, float32).")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of 1-minute audio chunks to process together (default: {DEFAULT_BATCH_SIZE}; lower if out of memory).",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "flash_attention_2", "sdpa"),
        default="auto",
        help="Attention backend; auto uses FlashAttention 2 when installed and falls back otherwise.",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    input_path = args.input_path
    processing_device = args.device
    processing_compute_type = args.compute_type

    if os.path.isfile(input_path):
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}
        if os.path.splitext(input_path.lower())[1] in video_extensions:
            print(f"\n{'='*60}")
            print(f"Processing single video: {os.path.basename(input_path)}")
            print(f"{'='*60}")
            
            base_name = os.path.splitext(input_path)[0]
            srt_path = f"{base_name}.srt"
            
            should_process = True
            if os.path.exists(srt_path):
                print(f"SRT file already exists: {srt_path}")
                user_input = input("Do you want to overwrite it? (y/n): ").lower().strip()
                if user_input != 'y':
                    print("Skipping this video...")
                    should_process = False
            
            if should_process:
                try:
                    generate_srt_from_video(
                        input_path,
                        output_dir=None,
                        device=processing_device,
                        compute_type=processing_compute_type,
                        batch_size=args.batch_size,
                        attention_implementation=args.attention_implementation,
                    )
                    print(f"✓ Successfully processed: {input_path}")
                except Exception as e:
                    print(f"✗ Error processing {input_path}: {str(e)}")
            
            print(f"\n{'='*60}")
            print("Processing complete!")
            print(f"{'='*60}")
        else:
             print(f"Error: Provided file '{input_path}' is not a recognized video format.")
    elif os.path.isdir(input_path):
        # Process all videos in the folder
        process_videos_in_folder(
            input_path,
            processing_device,
            processing_compute_type,
            batch_size=args.batch_size,
            attention_implementation=args.attention_implementation,
        )
    else:
        print(f"Error: Path '{input_path}' does not exist.")
