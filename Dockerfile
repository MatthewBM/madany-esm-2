# https://hub.docker.com/r/nvidia/cuda/tags?name=runtime
# OS/ARCH: linux/amd64
ARG OS_VERSION="22.04"
ARG CUDA_VERSION="12.1.1"
ARG ROOT_CONTAINER=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION}
FROM $ROOT_CONTAINER

LABEL maintainer="Matthew Madany <matthew.madany@gmail.com>"
ARG MAMBA_QUICK_BUILD=1
ARG TORCH_USER="ubuntu"
ARG TORCH_UID="1000"
ARG TORCH_GID="100"
ARG PYTHON_VERSION="3.10"
ARG PYTORCH_VERSION="2.2.0"
ARG PYTORCH_CUDA_VERSION="12.1"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# install root-level packages and ca-certificates
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    locales \
    ca-certificates \
    wget \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure build environment
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    TORCH_USER=$TORCH_USER \
    TORCH_UID=$TORCH_UID \
    TORCH_GID=$TORCH_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/$TORCH_USER
ENV PATH=$CONDA_DIR/bin:$PATH
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

COPY extras/fix-permissions /usr/local/bin/fix-permissions
RUN chmod +x /usr/local/bin/fix-permissions

# Setup conda with libraries for remote kernels
ENV PATH=$CONDA_DIR/bin:$PATH
RUN set -e; \
    wget --quiet "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -f -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh && \
    if [ "${MAMBA_QUICK_BUILD}" = "0" ]; then \
        BIN="${CONDA_DIR}/bin/conda"; \
    else \
        BIN="${CONDA_DIR}/bin/mamba"; \
    fi; \
    ln -s ${BIN} /usr/local/bin/conda-cmd && \
    ${BIN} install --yes \
        'jupyter_client' \
        'ipykernel' \
        'ipython' && \
    ${BIN} clean --all -f -y && \
    mkdir -p "/home/${TORCH_USER}" && \
    chown "${TORCH_UID}:${TORCH_GID}" "/home/${TORCH_USER}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${TORCH_USER}"

# Install and pin pytorch/cuda
RUN conda-cmd install --yes -c pytorch -c nvidia \
    "python=${PYTHON_VERSION}" \
    "pytorch=${PYTORCH_VERSION}" \
    "pytorch-cuda=${PYTORCH_CUDA_VERSION}" \
    && conda-cmd clean --all -f -y && \
    echo "python ${PYTHON_VERSION}.*" >> ${CONDA_DIR}/conda-meta/pinned && \
    echo "pytorch ${PYTORCH_VERSION}.*" >> ${CONDA_DIR}/conda-meta/pinned && \
    echo "pytorch-cuda ${PYTORCH_CUDA_VERSION}.*" >> ${CONDA_DIR}/conda-meta/pinned && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${TORCH_USER}"

# Consolidated install for ESM2 + our fastapi app
RUN conda-cmd install --yes -c conda-forge -c bioconda \
        'openmm' \
        'pdbfixer' \
        'einops' \
        'omegaconf' \
        'hydra-core' \
        'pandas' \
        'pytest' \
        'hmmer' \
        'hhsuite' \
        'kalign2' \
        'pip' \
        && conda-cmd clean --all -f -y && \
    "${CONDA_DIR}/bin/pip" install --no-cache-dir \
        biopython \
        deepspeed \
        dm-tree \
        fairscale \
        ml-collections \
        PyYAML \
        scipy \
        tqdm \
        pytorch_lightning \
        wandb \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        pydantic-settings \
        "transformers==4.44.2" \
        git+https://github.com/NVIDIA/dllogger.git \
      && fix-permissions "${CONDA_DIR}" && \
      fix-permissions "/home/${TORCH_USER}"

COPY --chown=${TORCH_UID}:${TORCH_GID} . /home/${TORCH_USER}

USER ${TORCH_UID}
WORKDIR /home/${TORCH_USER}
ENTRYPOINT ["python", "main.py"]
