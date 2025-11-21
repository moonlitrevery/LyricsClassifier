import argparse
from pathlib import Path
import json
import sys

# Garantir que o diretório do projeto (pai de scripts/) esteja no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyrics_classifier.config import TrainingConfig
from lyrics_classifier.corpus_loader import load_dataset
from lyrics_classifier.model_trainer import ModelTrainer


def parse_args():
    ap = argparse.ArgumentParser(description="Train lyrics classifier")
    ap.add_argument("--dataset", required=True, help="Caminho do dataset (CSV/XLSX/Parquet)")
    ap.add_argument("--text-col", default="musica")
    ap.add_argument("--label-col", default="genero")
    ap.add_argument("--language", default="pt", choices=["pt", "en"]) 
    ap.add_argument("--use-embeddings", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--models", default="nb_multinomial,logreg,linear_svm,random_forest,xgboost")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--calibrate", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--export-onnx", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = TrainingConfig(
        language=args.language,
        use_embeddings=args.use_embeddings,
        embedding_model=args.embedding_model,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        test_size=args.test_size,
        calibrate=args.calibrate,
        export_onnx=args.export_onnx,
    )

    workdir = Path(args.workdir)
    X, y = load_dataset(args.dataset, args.text_col, args.label_col)

    trainer = ModelTrainer(cfg, workdir)
    result = trainer.train(X.tolist(), y.tolist())

    print("Best model:", result.model_name)
    print("Classes:", result.classes_)
    print("Artifact dir:", str(result.artifact_dir))
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
