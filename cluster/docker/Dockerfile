# CUDA 12.x for B200/H200/H100
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /storage/projects

# Base tools + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo rsync openssh-server ffmpeg libsm6 libxext6 htop screen \
    build-essential wget curl xz-utils ca-certificates git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev tk-dev && \
    rm -rf /var/lib/apt/lists/*

# Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz && \
    tar -xzf Python-3.12.7.tgz && cd Python-3.12.7 && \
    ./configure --enable-optimizations --with-ssl --prefix=/usr/local && \
    make -j"$(nproc)" && make install && cd .. && rm -rf Python-3.12.7*

# Venv + PyTorch cu124 + W&B
RUN python3.12 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision torchaudio && \
    /opt/venv/bin/pip install wandb

ENV PATH="/opt/venv/bin:${PATH}" \
    WANDB_DIR=/storage/projects/.wandb \
    WANDB_CACHE_DIR=/storage/projects/.wandb/cache

RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys
COPY bin/* /usr/local/bin/
COPY sshd_config /etc/ssh/sshd_config
COPY .screenrc /root/.screenrc
RUN chmod 700 /usr/local/bin/ssh-start

EXPOSE 22
ENTRYPOINT ["ssh-start"]
CMD ["ssh-server"]
