# syntax=docker/dockerfile:1.6
#
# JupyterLab + PyTorch + PEFT for LoRA fine-tuning inside a Tinfoil Container.
# The base image ships torch with CUDA 13; everything else is pinned below so the
# image digest in tinfoil-config.yml describes exactly this toolchain.
FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime@sha256:9c99fafa01edfaa3d16da8c209b38b5970bb6fd6e72725ef60efc901489f70c6

ARG SOURCE_REVISION=unversioned
ARG VERSION=unversioned

RUN pip install --no-cache-dir \
      "transformers==5.17.0" \
      "peft==0.20.0" \
      "accelerate==1.15.0" \
      "jupyterlab==4.6.3" \
      "ipywidgets==8.1.9" \
      "matplotlib==3.11.2" \
 && python -c "import torch, transformers, peft, jupyterlab; print(torch.__version__, transformers.__version__, peft.__version__)"

# The notebook and sample data are seeded into the encrypted workspace on first start.
COPY notebook/finetune.ipynb /opt/example/finetune.ipynb
COPY data/train.jsonl data/eval.jsonl /opt/example/data/
COPY entrypoint.sh /opt/example/entrypoint.sh
RUN chmod 0755 /opt/example/entrypoint.sh

# The root filesystem is read-only in the enclave; every writable path is a tmpfs
# or the workspace volume (see tinfoil-config.yml).
ENV HOME=/root \
    JUPYTER_CONFIG_DIR=/tmp/jupyter/config \
    JUPYTER_DATA_DIR=/tmp/jupyter/data \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter/runtime \
    IPYTHONDIR=/tmp/ipython \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.source="https://github.com/tinfoilsh/finetuning-example" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${VERSION}"

EXPOSE 8888
ENTRYPOINT ["/opt/example/entrypoint.sh"]
