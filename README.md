# Lyrics Classifier – Pipeline de ML para Classificação de Letras

Projeto completo para classificar letras de músicas em categorias (gênero, humor/valência, década, tema). Inclui: coleta/ingestão, pré-processamento, extração de features (TF-IDF e embeddings), treino/validação, exportação do componente (joblib/ONNX) e serviço FastAPI.

- Sem Docker
- Reprodutível (seed, VERSION, artefatos versionados)
- Consumo local (artefatos) e remoto (API HTTP)

## Estrutura

```
lyrics-classifier/
  data/
    sample_lyrics.csv
  lyrics_classifier/
    __init__.py
    version.py
    config.py
    corpus_loader.py
    text_preprocessor.py
    feature_extractor.py
    model_trainer.py
    evaluator.py
    component.py
  service/
    app.py
  scripts/
    download_nltk.py
    train.py
    predict.py
    evaluate.py
  docs/
    report.md
    use_cases.md
    class_diagram.md
    sequence_diagrams.md
    component_diagram.md
    deployment_diagram.md
  reports/  (gerado após treino)
  artifacts/ (modelos exportados)
  configs/
    default.yaml
  requirements.txt
  VERSION
  README.md
```

## Requisitos e instalação

1) Crie venv e instale dependências

- Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python scripts\download_nltk.py
```

- Linux/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python scripts/download_nltk.py
```

Observação: `sentence-transformers` (embeddings) e `xgboost` podem demorar a instalar. Se necessário, rode só TF-IDF primeiro (veja flags abaixo).

## Dataset

- CSV com colunas: `lyrics` (texto) e `label` (classe). Exemplo em `data/sample_lyrics.csv`.

## Treinamento e validação

```
# TF-IDF (padrão), hold-out, escolhe melhor por F1-macro
python scripts/train.py \
  --dataset data/sample_lyrics.csv \
  --text-col lyrics --label-col label \
  --language pt \
  --models nb_multinomial,logreg,linear_svm,random_forest,xgboost \
  --test-size 0.2 \
  --calibrate True \
  --export-onnx False

# Com embeddings (sentence-transformers + LR)
python scripts/train.py \
  --dataset data/sample_lyrics.csv \
  --text-col lyrics --label-col label \
  --language pt \
  --use-embeddings True \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --models logreg \
  --test-size 0.2
```

Saídas em `artifacts/<modelo>_<timestamp>/` e métricas/figuras em `reports/`.

## Inferência local (componente)

```
# Usando último artefato automaticamente
python scripts/predict.py --texts "amo o som da guitarra" "batida que não para" --top-k 3

# Ou especificando o diretório do artefato
python scripts/predict.py --artifact-dir artifacts/<SEU_DIRETORIO> --texts "letra triste e lenta"
```

## Serviço FastAPI (API HTTP)

1) Defina qual artefato carregar (opcional):
```
$env:MODEL_DIR = "artifacts/<SEU_DIRETORIO>"   # Windows PowerShell
# export MODEL_DIR=artifacts/<SEU_DIRETORIO>   # Linux/macOS
```

2) Suba a API:
```
uvicorn service.app:app --reload --port 8000
```

3) Teste:
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["letra feliz animada","versos sombrios e lentos"],"top_k":3}'
```

## Reprodutibilidade

- `VERSION` do projeto é copiado dentro do artefato
- Config/supervisão salva em `config.json` no artefato
- Seeds fixos (`random_state`)
- Relatórios em `reports/` com métricas e matriz de confusão

## Documentação e diagramas

- Diagramas em Mermaid dentro de `docs/*.md` (Use Case, Classes, Sequência A/B, Componentes, Implantação)
- Para PDF: abra `docs/report.md` em um visualizador que exporte para PDF (VSCode + extensão Markdown PDF, ou `pandoc`).

## Tabela de versões

| Versão | Data | Notas |
|--------|------|-------|
| 0.1.0  | 2025-11-01 | Primeira entrega: pipeline, TF-IDF, embeddings opcionais, FastAPI, diagramas. |
