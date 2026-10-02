from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "face_clf.joblib"
REPORT_PATH = ROOT / "models" / "report.txt"
DELIVERIES_CSV = ROOT / "data" / "deliveries.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
UNKNOWN_THRESHOLD = 0.60  # below this probability -> "Unknown"

# folder name -> name used in deliveries.csv (extend as you add players)
STATS_NAME_MAP = {
    "Virat_Kohli": "V Kohli",
    "MS_Dhoni": "MS Dhoni",
    "Rohit_Sharma": "RG Sharma",
    "Shikhar_Dhawan": "S Dhawan",
    "Suresh_Raina": "SK Raina",
    "Ajinkya_Rahane": "AM Rahane",
    "Jasprit_Bumrah": "JJ Bumrah",
    "Ravindra_Jadeja": "RA Jadeja",
}
