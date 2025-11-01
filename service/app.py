from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from pathlib import Path
from lyrics_classifier.component import LyricsClassifier, find_latest_artifact
from starlette.responses import RedirectResponse, Response


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de letras para classificar")
    top_k: int = Field(3, ge=1, le=10)


app = FastAPI(title="Lyrics Classifier API", version="1.0.0")

_model: Optional[LyricsClassifier] = None


def _load_model() -> LyricsClassifier:
    global _model
    if _model is not None:
        return _model
    model_dir_env = os.getenv("MODEL_DIR")
    if model_dir_env:
        p = Path(model_dir_env)
        if not p.exists():
            raise RuntimeError(f"MODEL_DIR não encontrado: {p}")
        _model = LyricsClassifier.load(p)
    else:
        latest = find_latest_artifact(Path(__file__).resolve().parent.parent / "artifacts")
        if latest is None:
            raise RuntimeError("Nenhum artefato encontrado. Treine um modelo primeiro.")
        _model = LyricsClassifier.load(latest)
    return _model


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid 404 for favicon; return empty response."""
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metadata")
async def metadata():
    m = _load_model()
    return {
        "version": m.metadata.get("version"),
        "model_name": m.metadata.get("model_name"),
        "classes": m.classes_,
        "artifact_dir": m.metadata.get("artifact_dir"),
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    m = _load_model()
    if not req.texts:
        raise HTTPException(status_code=400, detail="Campo 'texts' vazio")
    topk = m.topk(req.texts, k=req.top_k)
    preds = m.predict(req.texts)
    return {
        "predictions": preds,
        "topk": topk,
        "classes": m.classes_,
        "version": m.metadata.get("version"),
    }
