# Use an official PyTorch image with GPU support
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

# Set the working directory
WORKDIR /app

# Install system dependencies (git, build tools, python3-dev, numpy-dev, etc.)
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    python3-dev \
    python3-numpy \
    wget \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the joint-disfluency-detector-and-parser repository
RUN git clone https://github.com/pariajm/joint-disfluency-detector-and-parser /app/joint-disfluency-detector-and-parser

# Build evalb using make
RUN cd /app/joint-disfluency-detector-and-parser/EVALB && \
    make evalb

# Copy the local directories to the container (src, swbd-data, results, viz, EVALB)
COPY --chown=daemon:daemon src/ /app/src
COPY --chown=daemon:daemon ../swbd-data /app/swbd-data
COPY --chown=daemon:daemon ../results /app/results
COPY --chown=daemon:daemon ../viz /app/viz
COPY --chown=daemon:daemon ../EVALB /app/EVALB

# Create a Python virtual environment and install dependencies (without PyTorch)
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    # Install numpy and cython first to avoid compilation issues
    /app/venv/bin/pip install numpy==1.23.3 cython --no-cache-dir && \
    # Install dependencies from requirements.txt (without PyTorch)
    /app/venv/bin/pip install -r /app/src/requirements.txt

# Set environment variables for your S3 bucket and other paths
ENV S3_BUCKET_NAME="com.trebble.ml.training.data"

# Ensure that PyTorch with CUDA is working correctly (This is a runtime check, not build-time)
RUN /app/venv/bin/python -c "import torch; print(torch.cuda.is_available())"

RUN cd /app/EVALB && \
    make evalb
# Make sure your working directory is correct for running main_improved.py
WORKDIR /app/src

# Use the virtual environment's Python to run the training script
CMD ["/app/venv/bin/python", "/app/src/train_parser.py", \
     "--config", "results/mobilebert_config.json", \
     "--override", '{"batch_size": 2, "learning_rate": 0.001}', \
     "--run-name", "mobilebert-disfluency-v1"]