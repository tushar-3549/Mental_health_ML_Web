from pathlib import Path
import textwrap, json, shutil, zipfile

base = Path("/mnt/data/mental_health_colab_integrated")
if base.exists():
    shutil.rmtree(base)
(base/"ml_training").mkdir(parents=True)
(base/"ml_artifacts").mkdir()
(base/"dataset").mkdir()

train = r'''
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "dataset" / "mental_health.csv"
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "Mental_Health_Condition"

FEATURES = [
    "Gender", "Age", "Academic_Year", "Sleep_Hours",
    "Social_Media_Usage", "Exercise", "Smoking",
    "Academic_Pressure", "Academic_Satisfaction", "Career_Concern",
    "Financial_Stress", "Family_Pressure", "Feeling_Depressed",
    "Feeling_Anxious", "Feeling_Stressed", "Difficulty_Concentrating",
    "Feeling_Lonely", "Feeling_Isolated"
]

def encode_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Remove accidental CSV index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Match the Colab encoding supplied by the user.
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["Exercise"] = df["Exercise"].map({"No": 0, "Yes": 1})
    df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})

    df["Academic_Year"] = df["Academic_Year"].map({
        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4
    })

    three_level = {"Low": 1, "Medium": 2, "High": 3}

    ordinal_cols = [
        "Social_Media_Usage", "Academic_Pressure",
        "Academic_Satisfaction", "Career_Concern",
        "Financial_Stress", "Family_Pressure",
        "Feeling_Depressed", "Feeling_Anxious",
        "Feeling_Stressed", "Difficulty_Concentrating",
        "Feeling_Lonely", "Feeling_Isolated"
    ]

    for col in ordinal_cols:
        df[col] = df[col].map(three_level)

    df[TARGET] = df[TARGET].map({
        "Normal": 0,
        "Moderate Risk": 1,
        "High Risk": 2
    })

    return df

def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET}\n\n"
            "Put your CSV at dataset/mental_health.csv"
        )

    df = pd.read_csv(DATASET)
    df = encode_dataset(df)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].dropna()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Same basic train/test structure expected by the supplied Colab evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(
            probability=True,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="mlogloss"
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = {
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )),
            "Recall": float(recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )),
            "F1 Score": float(f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ))
        }

        results.append(result)
        trained_models[name] = model

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, ARTIFACT_DIR / filename)

        print(
            f"  Accuracy={result['Accuracy']:.4f} | "
            f"Precision={result['Precision']:.4f} | "
            f"Recall={result['Recall']:.4f} | "
            f"F1={result['F1 Score']:.4f}"
        )

    results.sort(key=lambda x: x["F1 Score"], reverse=True)
    best_name = results[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")

    # Web application model comparison data.
    (ARTIFACT_DIR / "model_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    (ARTIFACT_DIR / "feature_config.json").write_text(
        json.dumps({
            "features": FEATURES,
            "target": TARGET,
            "best_model": best_name,
            "selection_metric": "F1 Score (macro)",
            "classes": {
                "Normal": 0,
                "Moderate Risk": 1,
                "High Risk": 2
            },
            "encoding": {
                "Gender": {"Female": 0, "Male": 1},
                "Exercise": {"No": 0, "Yes": 1},
                "Smoking": {"No": 0, "Yes": 1},
                "Academic_Year": {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4},
                "Low": 1,
                "Medium": 2,
                "High": 3
            }
        }, indent=2),
        encoding="utf-8"
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nBest model:", best_name)
    print("Saved:", ARTIFACT_DIR / "best_model.joblib")

if __name__ == "__main__":
    main()
'''

(base/"ml_training/train.py").write_text(textwrap.dedent(train), encoding="utf-8")

# Updated ML service compatible with the numeric encoding.
service = r'''
from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
CONFIG_PATH = ARTIFACT_DIR / "feature_config.json"

class MLService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "best_model.joblib not found. Run: python ml_training/train.py"
            )

        self.model = joblib.load(MODEL_PATH)

        self.metrics_data = []
        if METRICS_PATH.exists():
            self.metrics_data = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )

    @staticmethod
    def encode_input(data):
        row = dict(data)

        row["Gender"] = {"Female": 0, "Male": 1}[row["Gender"]]
        row["Exercise"] = {"No": 0, "Yes": 1}[row["Exercise"]]
        row["Smoking"] = {"No": 0, "Yes": 1}[row["Smoking"]]

        row["Academic_Year"] = {
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4
        }[row["Academic_Year"]]

        level = {"Low": 1, "Medium": 2, "High": 3}

        for field in [
            "Social_Media_Usage", "Academic_Pressure",
            "Academic_Satisfaction", "Career_Concern",
            "Financial_Stress", "Family_Pressure",
            "Feeling_Depressed", "Feeling_Anxious",
            "Feeling_Stressed", "Difficulty_Concentrating",
            "Feeling_Lonely", "Feeling_Isolated"
        ]:
            row[field] = level[row[field]]

        return row

    def predict(self, data):
        encoded = self.encode_input(data)
        row = pd.DataFrame([encoded])

        prediction = int(self.model.predict(row)[0])

        labels = {
            0: "Normal",
            1: "Moderate Risk",
            2: "High Risk"
        }

        confidence = None
        probabilities = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            confidence = float(max(probabilities) * 100)

        return {
            "class_id": prediction,
            "prediction": labels[prediction],
            "confidence": round(confidence, 2) if confidence is not None else None,
            "probabilities": probabilities.tolist()
                if probabilities is not None else None
        }

    def metrics(self):
        return self.metrics_data

_service = None

def get_ml_service():
    global _service
    if _service is None:
        _service = MLService()
    return _service
'''
(base/"ml_service_updated.py").write_text(textwrap.dedent(service), encoding="utf-8")

(base/"ml_artifacts/README.txt").write_text(
    "Run python ml_training/train.py to generate best_model.joblib, "
    "model_metrics.json and feature_config.json. Do not create fake model artifacts.\n",
    encoding="utf-8"
)
(base/"dataset/PUT_CSV_HERE.txt").write_text(
    "Copy your actual mental_health.csv into this folder.\n",
    encoding="utf-8"
)

zip_path = Path("/mnt/data/mental_health_colab_integrated_files.zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in base.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(base))

print(zip_path)

