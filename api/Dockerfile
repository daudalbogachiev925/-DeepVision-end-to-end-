FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi==0.111.0 uvicorn[standard]==0.30.1 \
    mlflow==2.13.0 boto3==1.34.0 pillow==10.3.0 \
    prometheus-client==0.20.0 python-multipart==0.0.9 \
    redis==5.0.4
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
