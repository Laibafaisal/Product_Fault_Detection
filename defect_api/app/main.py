import io, json, logging, os, time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("defect-api")

# ---------- Config (can be overridden with environment variables) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")
META_PATH = os.getenv("META_PATH", "models/meta.json")
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "5"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}

# ---------- Load model once at startup ----------
meta = json.load(open(META_PATH))
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
MEAN = np.array(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
STD = np.array(meta["std"], dtype=np.float32).reshape(1, 1, 3)
logger.info("Model loaded: %s (threshold=%s)", meta["model"], meta["threshold"])

app = FastAPI(title="Casting Defect Detection API", version="1.0.0")


def preprocess(image: Image.Image) -> np.ndarray:
    """Same steps as training: RGB -> resize -> scale 0-1 -> normalise -> NCHW."""
    size = meta["img_size"]
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model"], "threshold": meta["threshold"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    # 1) Validate the request
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Use JPEG/PNG/BMP.")
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB).")

    # 2) Decode the image
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        logger.warning("Could not decode file: %s", file.filename)
        raise HTTPException(400, "File is not a valid image.")

    # 3) Run the model
    try:
        logits = session.run(None, {"input": preprocess(image)})[0]
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, "Inference failed.")

    p_defect = float(softmax(logits)[0, 1])
    is_defect = p_defect >= meta["threshold"]
    confidence = p_defect if is_defect else 1.0 - p_defect

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    label = "defective" if is_defect else "normal"
    logger.info("file=%s pred=%s conf=%.3f latency=%sms", file.filename, label, confidence, latency_ms)

    return {"predicted_class": label,
            "confidence": round(confidence, 4),
            "defect_probability": round(p_defect, 4),
            "latency_ms": latency_ms}
