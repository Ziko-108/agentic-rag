# API image. Secrets are never copied in: the OpenAI key comes from .env at run time.
FROM python:3.12-slim

# CPU build of PyTorch by default (about 200 MB instead of ~3 GB for CUDA).
# docker-compose.gpu.yml overrides this with the CUDA index.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# The server runs as a normal user. The models folder is created for it now,
# not chown-ed later, because chown would copy the 1.1 GB model into another layer.
RUN useradd --create-home app && mkdir -p /opt/models && chown app /opt/models

# torch first, from its own index; requirements.txt then finds it already installed
RUN pip install --no-cache-dir torch==2.11.0 --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER app

# Download the models at build time, so the container starts without internet
# access to Hugging Face and the first question is not slow.
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANKER_MODEL}', allow_patterns=['*.json', '*.safetensors', '*.model'])"

# the models are in the image now: never ask the Hub again at start-up
ENV HF_HUB_OFFLINE=1

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
