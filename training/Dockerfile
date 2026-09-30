FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN pip install --no-cache-dir \
    pytorch-lightning==2.3.0 \
    torchmetrics==1.4.0 \
    mlflow==2.13.0 \
    boto3==1.34.0 \
    pillow==10.3.0 \
    timm==1.0.7 \
    albumentations==1.4.10 \
    opencv-python-headless==4.10.0.84

COPY . .
CMD ["python", "train.py"]
