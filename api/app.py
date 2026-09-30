  """FastAPI инференс-сервис для классификатора изображений."""
import io, os, json, hashlib, logging
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
import mlflow.pytorch
import redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

log = logging.getLogger("api"); logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DeepVision API", version="1.0.0")
cache = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, decode_responses=True)

# Prometheus
REQ = Counter("dv_requests_total", "Total requests", ["endpoint", "status"])
LAT = Histogram("dv_latency_seconds", "Latency", ["endpoint"])

_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model = None

def load():
    global model
    uri = os.getenv("MODEL_URI", "models:/deepvision-classifier/Production")
    log.info("Loading %s", uri)
    model = mlflow.pytorch.load_model(uri, map_location="cpu")
    model.eval()

@app.on_event("startup")
def startup():
    try: load()
    except Exception as e: log.warning("load failed: %s", e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    with LAT.labels("predict").time():
        raw = await file.read()
        key = "img:" + hashlib.sha256(raw).hexdigest()
        if cached := cache.get(key):
            REQ.labels("predict", "200").inc()
            return JSONResponse(json.loads(cached))

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            x = _preprocess(img).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]
            top = torch.topk(probs, top_k)
            result = {
                "predictions": [
                    {"class": CLASSES[i], "confidence": float(p)}
                    for p, i in zip(top.values, top.indices)
                ]
            }
            cache.setex(key, 3600, json.dumps(result))
            REQ.labels("predict", "200").inc()
            return result
        except Exception as e:
            REQ.labels("predict", "500").inc()
            raise HTTPException(500, str(e))
