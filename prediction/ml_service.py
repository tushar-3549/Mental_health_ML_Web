# from pathlib import Path
# import json
# import joblib
# import pandas as pd

# BASE_DIR = Path(__file__).resolve().parent.parent
# ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

# MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
# CONFIG_PATH = ARTIFACT_DIR / "feature_config.json"
# METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

# class MLService:
#     def __init__(self):
#         if not MODEL_PATH.exists():
#             raise FileNotFoundError(
#                 "ML model not found. Put your CSV in dataset/mental_health.csv and run: "
#                 "python ml_training/train.py"
#             )
#         self.model = joblib.load(MODEL_PATH)
#         self.config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

#     def predict(self, data):
#         row = pd.DataFrame([data])
#         prediction = self.model.predict(row)[0]

#         probabilities = None
#         confidence = None
#         if hasattr(self.model, "predict_proba"):
#             probabilities = self.model.predict_proba(row)[0]
#             confidence = float(max(probabilities) * 100)

#         return str(prediction), confidence, probabilities

#     def metrics(self):
#         if not METRICS_PATH.exists():
#             return []
#         return json.loads(METRICS_PATH.read_text())

# _service = None

# def get_ml_service():
#     global _service
#     if _service is None:
#         _service = MLService()
#     return _service




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
                "ML model not found. "
                "Run: python ml_training/train.py"
            )

        self.model = joblib.load(MODEL_PATH)

        if CONFIG_PATH.exists():
            self.config = json.loads(
                CONFIG_PATH.read_text()
            )
        else:
            self.config = {}

    def preprocess(self, data):
        """
        Convert API input into the same numerical representation
        used during ML model training.
        """

        encoded = data.copy()

        # -------------------------
        # Binary encoding
        # -------------------------

        gender_mapping = {
            "Female": 0,
            "Male": 1,
        }

        exercise_mapping = {
            "No": 0,
            "Yes": 1,
        }

        smoking_mapping = {
            "No": 0,
            "Yes": 1,
        }

        academic_year_mapping = {
            "1st": 1,
            "2nd": 2,
            "3rd": 3,
            "4th": 4,
        }

        encoded["Gender"] = gender_mapping.get(
            encoded["Gender"],
            encoded["Gender"],
        )

        encoded["Exercise"] = exercise_mapping.get(
            encoded["Exercise"],
            encoded["Exercise"],
        )

        encoded["Smoking"] = smoking_mapping.get(
            encoded["Smoking"],
            encoded["Smoking"],
        )

        encoded["Academic_Year"] = academic_year_mapping.get(
            encoded["Academic_Year"],
            encoded["Academic_Year"],
        )

        # -------------------------
        # Ordinal encoding
        # -------------------------

        three_level = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        ordinal_columns = [
            "Social_Media_Usage",
            "Academic_Pressure",
            "Academic_Satisfaction",
            "Career_Concern",
            "Financial_Stress",
            "Family_Pressure",
            "Feeling_Depressed",
            "Feeling_Anxious",
            "Feeling_Stressed",
            "Difficulty_Concentrating",
            "Feeling_Lonely",
            "Feeling_Isolated",
        ]

        for column in ordinal_columns:

            encoded[column] = three_level.get(
                encoded[column],
                encoded[column],
            )

        return encoded

    def predict(self, data):

        # Apply same preprocessing used during training
        encoded_data = self.preprocess(data)

        # Create DataFrame
        row = pd.DataFrame([encoded_data])

        # Predict
        prediction = self.model.predict(row)[0]

        # Probability / confidence
        probabilities = None
        confidence = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(row)[0]

            confidence = float(
                max(probabilities) * 100
            )

        return prediction, confidence, probabilities

    def metrics(self):

        if not METRICS_PATH.exists():
            return []

        return json.loads(
            METRICS_PATH.read_text()
        )


_service = None


def get_ml_service():

    global _service

    if _service is None:
        _service = MLService()

    return _service
