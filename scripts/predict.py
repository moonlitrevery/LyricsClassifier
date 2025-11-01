import argparse
from pathlib import Path
import sys

# Garantir que o diretório do projeto (pai de scripts/) esteja no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyrics_classifier.component import LyricsClassifier, find_latest_artifact


def parse_args():
    ap = argparse.ArgumentParser(description="Predict with lyrics classifier")
    ap.add_argument("--artifact-dir", default=None, help="Diretório do artefato (opcional)")
    ap.add_argument("--texts", nargs="+", required=True, help="Textos para classificar")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    return ap.parse_args()


def main():
    args = parse_args()
    if args.artifact_dir:
        art = Path(args.artifact_dir)
    else:
        latest = find_latest_artifact(Path(args.workdir) / "artifacts")
        if latest is None:
            print("Nenhum artefato encontrado. Treine um modelo primeiro.")
            return
        art = latest
    clf = LyricsClassifier.load(art)
    topk = clf.topk(args.texts, k=args.top_k)
    preds = clf.predict(args.texts)

    for i, t in enumerate(args.texts):
        print(f"\nTexto: {t}")
        print(f"Predição: {preds[i]}")
        print("Top-k:")
        for item in topk[i]:
            print(f"  {item['label']}: {item['prob']:.3f}")


if __name__ == "__main__":
    main()
