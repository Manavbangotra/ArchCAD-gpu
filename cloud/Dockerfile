# Training image for the line-based symbol spotter (vecformer/) on rented NVIDIA GPUs.
#
# Pins follow VecFormer's README: Python 3.9, torch 2.5.1 + CUDA 11.8, flash-attn
# built from source, spconv-cu118 2.3.8, torch-scatter. A -devel base is needed
# because flash-attn compiles against the CUDA toolkit.
#
#   docker build -t archcad-vecformer -f cloud/Dockerfile .
#   docker run --gpus all --shm-size=64g -v $PWD:/workspace -v /data:/data -it archcad-vecformer
#
# Build time is dominated by flash-attn (set MAX_JOBS to the build machine's cores;
# each job needs ~4 GB RAM).

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG MAX_JOBS=16
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 poppler-utils awscli \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge, no Anaconda licence terms) with Python 3.9.
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda install -y python=3.9 && conda clean -afy

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html \
    && pip install --no-cache-dir packaging psutil

# flash-attn from source, as VecFormer recommends (prebuilt wheels are tied to
# exact torch/CUDA/ABI combinations).
RUN git clone --depth 1 --branch v2.7.4.post1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa \
    && cd /tmp/fa && python setup.py install && cd / && rm -rf /tmp/fa

COPY vecformer/requirements.txt /tmp/vecformer-requirements.txt
COPY cloud/requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install --no-cache-dir -r /tmp/vecformer-requirements.txt -r /tmp/requirements-extra.txt

WORKDIR /workspace
CMD ["bash"]
