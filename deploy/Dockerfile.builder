FROM docker.io/nvidia/cuda:13.3.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    git \
    rsync \
    file \
    python3 \
    python3-jinja2 \
    python3-pip \
    zlib1g-dev \
    libcurl4-openssl-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages gguf numpy torch
