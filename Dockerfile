# 1. Start with NVIDIA CUDA 12.1 runtime
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python 3.8 and system essentials
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git curl libgl1-mesa-glx libglib2.0-0 \
    python3.8 python3.8-dev python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Python 3.8 as default and install Pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
# Use the specific legacy script for Python 3.8
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && python get-pip.py

# 5. Install PyTorch 2.2.2 (The "Big" dependency)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Install NVIDIA Warp (Required for robot physics)
RUN pip install warp-lang==1.8.0

# 7. Install the rest of the requirements
# We do this specifically so Docker caches the steps above
RUN pip install --no-cache-dir \
    tqdm==4.67.1 \
    pyglet==2.1.6 \
    ipdb \
    h5py==3.11.0 \
    pyyaml==6.0.2 \
    tensorboard==2.14.0 \
    matplotlib==3.7.5 \
    opencv-python \
    pycollada==0.9.2 \
    scipy==1.10.1 \
    trimesh==4.7.1 \
    wandb \
    rl-games

# 8. Set working directory
WORKDIR /playground

CMD ["/bin/bash"]