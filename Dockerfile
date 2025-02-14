# Base image with GPU support and PyTorch
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set environment variables for Python and CUDA
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python libraries
RUN pip install --no-cache-dir \
    langchain_community \
    tiktoken \
    langchain-openai \
    langchainhub \
    chromadb \
    youtube-transcript-api \
    pytube \
    ragatouille \
    transformers \
    torch \
    torchvision \
    torchaudio \
    diffusers \
    numpy \
    matplotlib \
    opencv-python \
    pandas \
    keybert \
    ctransformers[cuda] \
    python-dotenv 

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -U sentence-transformers==2.2.2
RUN pip install langchain-huggingface
RUN pip install -U langchain-community
RUN pip install 'accelerate>=0.26.0'
RUN pip install pydub
RUN pip install yt-dlp

RUN apt-get install libsndfile1

RUN pip install librosa soundfile

# Verify GPU availability in Python
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
