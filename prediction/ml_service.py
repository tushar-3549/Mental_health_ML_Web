from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "feature_config.json"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

class MLService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "ML model not found. Put your CSV in dataset/mental_health.csv and run: "
                "python ml_training/train.py"
            )
        self.model = joblib.load(MODEL_PATH)
        self.config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

    def predict(self, data):
        row = pd.DataFrame([data])
        prediction = self.model.predict(row)[0]

        probabilities = None
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            confidence = float(max(probabilities) * 100)

        return str(prediction), confidence, probabilities

    def metrics(self):
        if not METRICS_PATH.exists():
            return []
        return json.loads(METRICS_PATH.read_text())

_service = None

def get_ml_service():
    global _service
    if _service is None:
        _service = MLService()
    return _service
