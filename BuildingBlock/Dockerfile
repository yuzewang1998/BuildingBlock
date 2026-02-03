FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-distutils \
    python3.8-dev \
    git \
    xvfb \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    python3-tk


RUN ln -sf /usr/bin/python3.8 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118


RUN pip install cython==0.29.32 \
    numpy==1.24.4 \
    networkx==2.5.1 \
    pyrr==0.10.3 \
    pyyaml==5.3.1 \
    pillow==10.4.0 \
    scipy==1.8.1 \
    tqdm==4.64.1 \
    matplotlib==3.4.1 \
    trimesh==3.12.7 \
    simple_3dviz==0.4.0 \
    pytorch-fast-transformers==0.4.0 \
    kornia==0.6.9 \
    transformers==4.25.1 \
    pyvista==0.43.3 \
    ftfy \
    regex \
    seaborn==0.13.2 \
    torchtext==0.6.0 \
    nltk \
    num2words==0.5.13 \
    swanlab \
    wandb \
    clean-fid \
    einops 

RUN pip install git+https://github.com/openai/CLIP.git


