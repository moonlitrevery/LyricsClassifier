# Diagrama de Componentes e Contratos

```mermaid
flowchart TB
  subgraph ModelComponent["Componente de Modelo"]
    ART[("component.joblib")]
    CFG[["config.json"]]
    VER[["VERSION"]]
    MET[["metrics.json"]]
    ONNX[["model.onnx (opcional)"]]
  end

  subgraph ServiceComponent["Componente de Serviço (API)"]
    API[["FastAPI /predict, /metadata, /health"]]
  end

  subgraph Client["Consumidores"]
    LIB["Cliente Local (Python package)"]
    HTTP["Cliente HTTP"]
  end

  LIB -->|"Carrega"| ART
  API -->|"Carrega"| ART
  API -.->|"Lê"| CFG
  API -.->|"Lê"| VER
  API -.->|"Lê"| MET

  LIB -->|"predict(texts), topk(k)"| ART
  HTTP -->|"POST /predict"| API
  API -->|"JSON Response"| HTTP
```

Interfaces e formatos principais:
- Componente Local (Python):
  - predict(texts: List[str]) -> List[str]
  - predict_proba(texts: List[str]) -> Optional[List[List[float]]]
  - topk(texts: List[str], k: int) -> List[List[{label, prob}]]
- API HTTP (FastAPI):
  - POST /predict
    - Request: { "texts": [string], "top_k": int }
    - Response: { "predictions": [string], "topk": [[{label, prob}]], "classes": [string], "version": string }
  - GET /metadata → {version, model_name, classes, artifact_dir}
  - GET /health → {status}
