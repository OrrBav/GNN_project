# Use NVIDIA CUDA base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Copy Miniconda installer from your host to the container
COPY Miniconda3-latest-Linux-x86_64.sh /miniconda.sh

# Install Miniconda
RUN bash /miniconda.sh -b -p /opt/conda && rm /miniconda.sh

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "--login", "-c"]

# Add a non-root user and ensure it has full access to Conda
RUN useradd -m -s /bin/bash myuser && \
    mkdir -p /home/myuser/.cache/conda && \
    mkdir -p /opt/conda/pkgs_cache && \
    chown -R myuser /opt/conda /home/myuser/.cache /opt/conda/pkgs_cache

# Switch to the new user
USER myuser
WORKDIR /home/myuser

# Create Conda environment & Install FAISS GPU + Other Dependencies
RUN conda create -n faiss_env python=3.9 -y && \
    conda run -n faiss_env conda install -y -c pytorch -c nvidia \
        pytorch torchvision torchaudio pytorch-cuda=12.1 && \
    conda run -n faiss_env pip install \
        faiss-gpu-cu12 \
        pandas \
        numpy \
        networkx \
        matplotlib \
        scikit-learn

## Optional: you can install torch packages also:     conda run -n faiss_env conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

# Set environment to activate Conda automatically
ENV CONDA_ENV=faiss_env
ENV CONDA_DEFAULT_ENV=$CONDA_ENV
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH

# Start the container with Conda environment activated
CMD ["bash", "-c", "source activate faiss_env && exec bash"]
