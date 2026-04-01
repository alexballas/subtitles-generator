# Video Subtitle Generator

This script generates precise, aligned `.srt` subtitles for video files using `whisperx`.

## Prerequisites

1.  Ensure you have Python installed.
2.  Set up the virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

You can use the provided `run.sh` script to automatically activate the environment, set required library paths (for CUDA support), and run the generator. 

### Basic Usage

Process a single video file (the `.srt` will be saved next to the video):
```bash
./run.sh /path/to/your/video.mp4
```

Process an entire folder of videos:
```bash
./run.sh /path/to/folder/containing/videos
```

### Advanced Options

By default, the script attempts to use the GPU (`cuda`) and `float16` precision. You can override these options using command-line arguments:

**Run on CPU:**
```bash
./run.sh /path/to/video.mp4 --device cpu
```

**Change compute type (e.g., int8, float32, float16):**
```bash
./run.sh /path/to/video.mp4 --compute_type int8
```

View the help menu for all options:
```bash
./run.sh --help
```
