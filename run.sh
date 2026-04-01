#!/usr/bin/env bash

set -euo pipefail

source venv/bin/activate

python_version="python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
venv_lib_dir="$VIRTUAL_ENV/lib/$python_version/site-packages"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$venv_lib_dir/torch/lib:$venv_lib_dir/nvidia/cublas/lib:$venv_lib_dir/nvidia/cudnn/lib:$venv_lib_dir/nvidia/cuda_runtime/lib"
python main.py "$@"
