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
                        "language": detected_language,
                    })
                else:
                    segments.append({
                        "start": start_seconds,
                        "end": end_seconds,
                        "text": text,
                        "language": detected_language,
                    })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return segments, list(dict.fromkeys(detected_languages))


def _align_missing_segments(segments: List[Dict], audio, device: str) -> None:
    """Retry alignment for chunks that Qwen transcribed without word times."""
    by_language = {}
    for segment in segments:
        if segment.get("words"):
            continue
        language_code = _qwen_language_to_whisper_code(segment.get("language", ""))
        if language_code:
            by_language.setdefault(language_code, []).append(segment)
        else:
            print(f"No alignment language for chunk at {format_time_srt(segment['start'])}.")

    for language_code, missing in by_language.items():
        model_a = None
        try:
            print(f"Retrying word alignment for {len(missing)} {language_code} chunk(s)...")
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code, device=device
            )
            aligned = whisperx.align(
                missing, model_a, metadata, audio, device,
                return_char_alignments=False,
            )["segments"]
            for original, replacement in zip(missing, aligned):
                words = [
                    word for word in replacement.get("words", [])
                    if word.get("word") and word.get("start") is not None
                    and word.get("end") is not None
                    and word["end"] > word["start"]
                ]
                if words:
                    original["words"] = words
                    original["start"] = words[0]["start"]
                    original["end"] = words[-1]["end"]
        except Exception as exc:
            print(f"WhisperX alignment unavailable for {language_code}: {exc}")
        finally:
            del model_a
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


def _refine_suspicious_timing(segments: List[Dict], audio) -> None:
    """Use detected speech to correct implausible alignment timestamps."""
    long_word_seconds = 2.5
    isolated_gap_seconds = 2.5

    def may_lead_following_phrase(words: List[Dict], index: int) -> bool:
        word = words[index]["word"]
        following = words[index + 1]["word"]
        is_title = word.casefold() in {"mr.", "mrs.", "ms.", "dr."}
        return (
            (is_title or not word.endswith((".", "!", "?")))
            and bool(following)
            and (is_title or following[0].islower())
            and (
                is_title
                or index == 0
                or words[index - 1]["word"].endswith((".", "!", "?"))
                or words[index]["start"] - words[index - 1]["end"] > isolated_gap_seconds
            )
        )

    suspicious = any(
        not segment.get("words") or any(
            word["end"] - word["start"] > long_word_seconds
            for word in segment["words"]
        ) or any(
            segment["words"][index + 1]["start"] - word["end"] > isolated_gap_seconds
            and may_lead_following_phrase(segment["words"], index)
            for index, word in enumerate(segment["words"][:-1])
        )
        for segment in segments
    )
    if not suspicious:
        return

    try:
        from whisperx.vads.pyannote import Binarize, Pyannote

        print("Checking suspicious subtitle timings against detected speech...")
        vad = Pyannote(torch.device("cpu"), vad_onset=0.5)
        scores = vad({
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": QWEN_SAMPLE_RATE,
        })
        speech = Binarize(
            onset=0.5, offset=0.363,
            min_duration_on=0.1, min_duration_off=0.1,
        )(scores)
        speech_intervals = [(region.start, region.end) for region in speech.get_timeline()]
        del vad, scores
    except Exception as exc:
        print(f"Speech timing check unavailable: {exc}")
        return

    def speech_at(start: float, end: float):
        return next((
            (speech_start, speech_end)
            for speech_start, speech_end in speech_intervals
            if speech_start < end + 0.2 and speech_end > start - 0.2
        ), None)

    for segment in segments:
        words = segment.get("words")
        if words:
            original_durations = {
                id(word): word["end"] - word["start"] for word in words
            }
            for word in words:
                start, end = word["start"], word["end"]
                if end - start <= long_word_seconds:
                    continue
                speech_at_start = speech_at(start, min(start + 0.4, end))
                if speech_at_start:
                    # A later voice can occur inside a stretched word span. Keep
                    # the start when speech exists there and trim the silent tail.
                    corrected_end = min(end, speech_at_start[1] + 0.2)
                    if end - corrected_end >= 0.5 and corrected_end > start:
                        print(
                            f"Trimmed long word end at {format_time_srt(end)} "
                            f"to {format_time_srt(corrected_end)}: {word['word']}"
                        )
                        word["end"] = corrected_end
                    continue
                # The aligner sometimes stretches a word backward through silence.
                # Only move its start when speech is detected near its end.
                matching = [
                    speech_start for speech_start, speech_end in speech_intervals
                    if start < speech_start < end
                    and speech_end >= end - 0.25
                ]
                if matching:
                    corrected_start = max(start, matching[-1] - 0.15)
                    if corrected_start - start >= 0.5:
                        print(
                            f"Moved early word start at {format_time_srt(start)} "
                            f"to {format_time_srt(corrected_start)}: {word['word']}"
                        )
                        word["start"] = corrected_start

            for index, (word, following) in enumerate(zip(words, words[1:])):
                gap = following["start"] - word["end"]
                if gap <= isolated_gap_seconds or not may_lead_following_phrase(words, index):
                    continue
                speech_at_old_start = speech_at(word["start"], word["end"])
                # Speech at the old position might belong to a different line.
                # A word stretched across a long silence is more likely to lead
                # the following phrase, even if another voice was heard earlier.
                stretched_before_trim = original_durations[id(word)] > long_word_seconds
                if speech_at_old_start and stretched_before_trim and gap > 35.0:
                    print(
                        f"Review ambiguous word near {format_time_srt(word['start'])}: "
                        f"{word['word']} (next word is {gap:.1f}s later)"
                    )
                if speech_at_old_start and (
                    not stretched_before_trim or gap < 4.0 or gap > 35.0
                ):
                    continue
                following_speech = speech_at(following["start"], following["end"])
                if following_speech is None:
                    continue

                # A short word aligned in silence can be separated from its sentence
                # by a large gap. Anchor it near the speech that contains the next word.
                corrected_start = max(
                    following_speech[0] - 0.2,
                    following["start"] - 0.5,
                )
                corrected_end = max(following["start"], corrected_start + 0.12)
                print(
                    f"Moved isolated word from {format_time_srt(word['start'])} "
                    f"to {format_time_srt(corrected_start)}: {word['word']}"
                )
                word["start"] = corrected_start
                word["end"] = corrected_end
            segment["start"] = words[0]["start"]
            segment["end"] = words[-1]["end"]
        else:
            nearby = [
                (speech_start, speech_end)
                for speech_start, speech_end in speech_intervals
                if speech_end > segment["start"] and speech_start < segment["end"]
            ]
            if nearby:
                segment["start"] = max(segment["start"], nearby[0][0] - 0.15)
                segment["end"] = min(segment["end"], nearby[-1][1] + 0.25)
            else:
                print(
                    f"Could not locate speech for unaligned chunk at "
                    f"{format_time_srt(segment['start'])}; leaving its timing unchanged."
                )


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
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    detected_language = ",".join(detected_languages)
    print(f"Detected language(s): {detected_language or 'unknown'}")

    if not segments:
        print("No speech detected.")
        return

    _align_missing_segments(segments, audio, device)
    _refine_suspicious_timing(segments, audio)

    print(f"Generating SRT file: {output_srt_path}")
    subtitle_counter = 1
    all_sub_segments = []

    for segment in segments:
        all_sub_segments.extend(split_by_pauses(segment, min_pause=0.3, max_duration=5.0, max_chars=80))

    all_sub_segments = extend_subtitle_timing(all_sub_segments, linger_time=0.5, safety_gap=0.02)

    long_subtitles = [
        segment for segment in all_sub_segments
        if segment["end"] - segment["start"] > 6.0
    ]
    if long_subtitles:
        print(f"Review {len(long_subtitles)} subtitle(s) longer than 6 seconds:")
        for segment in long_subtitles:
            print(f"  {format_time_srt(segment['start'])}: {segment['text'][:60]}")
    
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
