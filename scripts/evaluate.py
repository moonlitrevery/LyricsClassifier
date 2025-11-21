import argparse
from pathlib import Path
import sys
from sklearn.metrics import classification_report

# Garantir que o diretório do projeto (pai de scripts/) esteja no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyrics_classifier.component import LyricsClassifier
from lyrics_classifier.corpus_loader import load_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate artifact on a labeled dataset")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--text-col", default="musica")
    ap.add_argument("--label-col", default="genero")
    return ap.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.dataset, args.text_col, args.label_col)

    clf = LyricsClassifier.load(Path(args.artifact_dir))
    preds = clf.predict(X.tolist())
    print(classification_report(y, preds, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
