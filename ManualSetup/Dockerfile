# Run this if devcontainer did not work
# Still requires some files from original TFLite-demo-container

# docker build \
#   -f Dockerfile \
#   -t eai-rpi-dev .

# docker run --rm -it \
#   -v "$PWD:/workspace" \
#   eai-rpi-dev 


FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake ninja-build \
    rsync openssh-client sshpass tmux \
    gdb-multiarch \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    git curl wget unzip ca-certificates \
    python3 python3-pip python3-dev python3.10-venv \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
USER vscode
