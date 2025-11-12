import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Конфиг
# -----------------------------
# Используем базовую модель mDeBERTa-v3 обученную на NLI
BASE_MODEL = os.getenv("BASE_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
THRESHOLD_CREDIBLE = float(os.getenv("THRESHOLD_CREDIBLE", "0.70"))
THRESHOLD_QUESTIONABLE = float(os.getenv("THRESHOLD_QUESTIONABLE", "0.40"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:4200"
).split(",")

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="KZ Fake News (mDeBERTa-v3 NLI-based)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Метки для NLI модели (entailment, neutral, contradiction)
# Мапим на наши классы: entailment -> True, contradiction -> False
# -----------------------------
LABELS = ["True", "Mixed", "False"]
IDX_TRUE = 0  # entailment
IDX_MIXED = 1  # neutral
IDX_FALSE = 2  # contradiction

id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}

# -----------------------------
# Загрузка токенайзера и модели
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
model.to(device).eval()

# -----------------------------
# Pydantic схемы
# -----------------------------
class PredictReq(BaseModel):
    text: str = Field(..., min_length=3, description="Новостной текст (KZ/RU/EN)")

class PredictBatchReq(BaseModel):
    items: List[PredictReq]

class PredictResp(BaseModel):
    probs: Dict[str, float]
    score: float
    verdict: str

# -----------------------------
# Вспомогательные
# -----------------------------
def softmax_preds(texts: List[str]) -> torch.Tensor:
    # Для NLI модели используем hypothesis "This is true information"
    premises = texts
    hypotheses = ["This is true and credible information."] * len(texts)
    
    enc = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
    
    # Преобразуем NLI выход (entailment, neutral, contradiction) в наши метки
    # entailment -> True, neutral -> Mixed, contradiction -> False
    batch_size = probs.shape[0]
    remapped = torch.zeros(batch_size, 3, device=probs.device)
    remapped[:, IDX_TRUE] = probs[:, 0]  # entailment
    remapped[:, IDX_MIXED] = probs[:, 1]  # neutral
    remapped[:, IDX_FALSE] = probs[:, 2]  # contradiction
    
    return remapped

def decision_from_ptrue(p_true: float) -> str:
    if p_true >= THRESHOLD_CREDIBLE:
        return "credible"
    if p_true >= THRESHOLD_QUESTIONABLE:
        return "questionable"
    return "fake"

# -----------------------------
# Эндпоинты
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "device": device,
        "labels": LABELS,
        "thresholds": {
            "credible": THRESHOLD_CREDIBLE,
            "questionable": THRESHOLD_QUESTIONABLE
        },
        "base_model": BASE_MODEL,
    }

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq):
    probs = softmax_preds([req.text]).squeeze(0).tolist()
    out = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    p_true = float(probs[IDX_TRUE])
    verdict = decision_from_ptrue(p_true)
    return {"probs": out, "score": p_true, "verdict": verdict}

@app.post("/predict-batch", response_model=List[PredictResp])
def predict_batch(req: PredictBatchReq):
    texts = [it.text for it in req.items]
    if not texts:
        return []
    probs_all = softmax_preds(texts).cpu().tolist()
    responses = []
    for vec in probs_all:
        out = {LABELS[i]: float(vec[i]) for i in range(len(LABELS))}
        p_true = float(vec[IDX_TRUE])
        verdict = decision_from_ptrue(p_true)
        responses.append({"probs": out, "score": p_true, "verdict": verdict})
    return responses

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)