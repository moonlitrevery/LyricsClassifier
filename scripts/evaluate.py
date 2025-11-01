import argparse
from pathlib import Path
import sys
from sklearn.metrics import classification_report
import pandas as pd

# Garantir que o diretório do projeto (pai de scripts/) esteja no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyrics_classifier.component import LyricsClassifier


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate artifact on a labeled dataset")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--text-col", default="lyrics")
    ap.add_argument("--label-col", default="label")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset)
    X = df[args.text_col].astype(str).tolist()
    y = df[args.label_col].astype(str).tolist()

    clf = LyricsClassifier.load(Path(args.artifact_dir))
    preds = clf.predict(X)
    print(classification_report(y, preds, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
