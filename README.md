# Indian Cricket Team Member Face Identification

Classifies a face photo into the matching cricketer's name.

**Pipeline:** MTCNN (detect + align) -> FaceNet/VGGFace2 (512-d embedding) -> linear SVM -> name + confidence.
Below `UNKNOWN_THRESHOLD` in `src/config.py` the result is "Unknown".
Optional extra: IPL career stats for the identified player from `data/deliveries.csv`.

## Structure
```
app.py              Streamlit demo
src/config.py       paths, threshold, folder-name -> IPL-name map
src/face_utils.py   detection + embeddings
src/train.py        build embeddings, train SVM, write models/report.txt
src/predict.py      CLI + predict() used by the app
src/stats.py        stats from deliveries.csv
dataset/            YOUR face images (one folder per player)
data/deliveries.csv IPL ball-by-ball data
```

## Setup
    pip install -r requirements.txt

## 1. Add images
    dataset/Virat_Kohli/001.jpg ...
    dataset/MS_Dhoni/001.jpg ...

50+ varied photos per player (angles, lighting, caps/beards), one clear face preferred.
Use images you have the right to use. Keep a few photos aside as a final test set.

## 2. Train
    python -m src.train

Prints per-player precision/recall and saves `models/face_clf.joblib`.

## 3. Predict
    python -m src.predict some_photo.jpg
    streamlit run app.py

## Notes
- Folder names use underscores (`Rohit_Sharma`). To show stats, map them to the
  deliveries.csv name in `STATS_NAME_MAP` (e.g. `RG Sharma`).
- Check `models/report.txt` and look at misclassified photos to improve the dataset.
- First run downloads the pretrained FaceNet weights (needs internet).
- Face recognition is sensitive: use it only with images you're permitted to use.
