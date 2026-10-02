"""Train: python -m src.train"""
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .config import DATASET_DIR, IMAGE_EXTS, MODEL_PATH, REPORT_PATH
from .face_utils import embed_image


def load_embeddings():
    X, y = [], []
    people = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if len(people) < 2:
        raise SystemExit("Add at least 2 player folders to dataset/ (see README).")
    for person in people:
        n_img = 0
        for p in sorted(person.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                embs = embed_image(Image.open(p))
            except Exception as e:
                print(f"skip {p.name}: {e}")
                continue
            for e in embs:
                X.append(e)
                y.append(person.name)
            n_img += bool(embs)
        print(f"{person.name}: {n_img} usable images")
    return np.array(X), np.array(y)


def main():
    X, y = load_embeddings()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    clf = SVC(kernel="linear", probability=True, C=1.0)
    clf.fit(X_tr, y_tr)
    report = classification_report(y_te, clf.predict(X_te), target_names=le.classes_)
    print(report)
    REPORT_PATH.write_text(report)

    clf.fit(X, y_enc)  # final model on all data
    joblib.dump({"clf": clf, "le": le}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
