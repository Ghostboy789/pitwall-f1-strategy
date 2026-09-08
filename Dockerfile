# One image, five hosts. Render, Fly, Railway, Koyeb and Hugging Face Spaces
# all run this unchanged -- the app binds $PORT and nothing else is host-specific.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the environment.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pitwall/ ./pitwall/
COPY app/ ./app/
# Fitted artefacts are committed for deployment: the container serves a model,
# it does not refit one. ~100 KB, including circuit_reference.parquet, which
# stands in for the 12.7 MB lap table the app would otherwise need.
COPY models_out/ ./models_out/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
