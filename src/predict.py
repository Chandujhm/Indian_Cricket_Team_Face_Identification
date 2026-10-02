"""Predict: python -m src.predict path/to/image.jpg"""
import sys
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image

from .config import MODEL_PATH, UNKNOWN_THRESHOLD
from .face_utils import embed_image


@lru_cache(maxsize=1)
def _bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict(img: Image.Image, top_k=3):
    """Returns (best_name_or_None, [(name, prob), ...]); best is None if low confidence."""
    embs = embed_image(img)
    if not embs:
        return None, []
    b = _bundle()
    probs = b["clf"].predict_proba([embs[0]])[0]
    order = np.argsort(probs)[::-1][:top_k]
    top = [(b["le"].classes_[i], float(probs[i])) for i in order]
    best = top[0][0] if top[0][1] >= UNKNOWN_THRESHOLD else None
    return best, top


if __name__ == "__main__":
    best, top = predict(Image.open(sys.argv[1]))
    if not top:
        print("No face detected")
    else:
        print("Prediction:", best or "Unknown / low confidence")
        for n, p in top:
            print(f"  {n.replace('_', ' ')}: {p:.1%}")
