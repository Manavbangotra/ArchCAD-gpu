#!/usr/bin/env bash
# Same environment as cloud/Dockerfile, for a bare Ubuntu GPU VM with a CUDA 11.8+
# driver (no Docker). Creates conda env "vecformer".
#
#   bash cloud/setup_vm.sh            # from the repository root
#   conda activate vecformer
set -euo pipefail
MAX_JOBS=${MAX_JOBS:-$(nproc)}

if ! command -v conda >/dev/null; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$HOME/miniforge3"
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
    eval "$(conda shell.bash hook)"
fi

conda create -y -n vecformer python=3.9
conda activate vecformer
# flash-attn compiles against the CUDA toolkit; take it from conda if the VM has none.
if ! command -v nvcc >/dev/null; then
    conda install -y -c nvidia cuda-toolkit=11.8
fi

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install packaging ninja psutil
if ! python -c "import flash_attn" 2>/dev/null; then
    rm -rf /tmp/fa && git clone --depth 1 --branch v2.7.4.post1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa
    (cd /tmp/fa && MAX_JOBS=$MAX_JOBS python setup.py install)
fi
pip install -r vecformer/requirements.txt -r cloud/requirements-extra.txt

python - <<'PY'
import torch, flash_attn, spconv.pytorch, torch_scatter
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("flash_attn", flash_attn.__version__, "| spconv ok | torch_scatter ok")
PY
