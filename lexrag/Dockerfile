# LexRAG: aplikacja z interfejsem WWW. Model językowy działa w osobnym kontenerze (Ollama).
# Domyślnie torch z CUDA (reranker i embeddingi na karcie; LEXRAG_DEVICE=auto). Obraz tylko na CPU:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t lexrag .
FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/modele LEXRAG_DATA_DIR=/app/data LEXRAG_INDEX_DIR=/app/data/index
RUN apt-get update && apt-get install -y --no-install-recommends pandoc curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-ml.txt requirements-web.txt ./
RUN pip install torch --index-url ${TORCH_INDEX} \
 && pip install numpy simplemma pymupdf -r requirements-ml.txt -r requirements-web.txt
COPY lexrag ./lexrag
COPY scripts ./scripts
COPY data/golden ./data/golden
EXPOSE 8090
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s CMD curl -fs http://localhost:8090/api/stan || exit 1
CMD ["uvicorn", "lexrag.web:app", "--host", "0.0.0.0", "--port", "8090"]
