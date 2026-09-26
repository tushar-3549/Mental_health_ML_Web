from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "mental_health.csv"
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)


TARGET = "Mental_Health_Condition"

FEATURES = [
    "Gender",
    "Age",
    "Academic_Year",
    "Sleep_Hours",
    "Social_Media_Usage",
    "Exercise",
    "Smoking",
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
    "Feeling_Isolated"
]


def encode_dataset(df):

    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    # Gender
    df["Gender"] = df["Gender"].map({
        "Female": 0,
        "Male": 1
    })

    # Exercise
    df["Exercise"] = df["Exercise"].map({
        "No": 0,
        "Yes": 1
    })

    # Smoking
    df["Smoking"] = df["Smoking"].map({
        "No": 0,
        "Yes": 1
    })

    # Academic Year
    df["Academic_Year"] = df["Academic_Year"].map({
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "4th": 4
    })

    # Low / Medium / High
    three_level = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    ordinal_cols = [
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
        "Feeling_Isolated"
    ]

    for col in ordinal_cols:
        df[col] = df[col].map(three_level)

    # Target
    df[TARGET] = df[TARGET].map({
        "Normal": 0,
        "Moderate Risk": 1,
        "High Risk": 2
    })

    return df


def main():

    print("Checking dataset...")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    print("Loading dataset...")

    df = pd.read_csv(DATASET)

    print("Original shape:", df.shape)

    df = encode_dataset(df)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[required_columns]

    df = df.dropna()

    print("After encoding/dropna:", df.shape)

    X = df[FEATURES]

    y = df[TARGET].astype(int)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    # Models
    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(),

        "SVM":
            SVC(
                probability=True,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                eval_metric="mlogloss"
            )
    }

    results = []

    trained_models = {}

    # Train models
    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        result = {
            "Model": name,
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1)
        }

        results.append(result)

        trained_models[name] = model

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

    # Sort according to F1
    results.sort(
        key=lambda x: x["F1 Score"],
        reverse=True
    )

    # Best model
    best_name = results[0]["Model"]

    best_model = trained_models[best_name]

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    print(
        pd.DataFrame(results).to_string(
            index=False
        )
    )

    print("\nBest model:", best_name)

    # Save best model
    joblib.dump(
        best_model,
        ARTIFACT_DIR / "best_model.joblib"
    )

    # Save metrics
    with open(
        ARTIFACT_DIR / "model_metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    # Save feature configuration
    feature_config = {

        "features": FEATURES,

        "target": TARGET,

        "best_model": best_name,

        "classes": {
            "Normal": 0,
            "Moderate Risk": 1,
            "High Risk": 2
        },

        "encoding": {

            "Gender": {
                "Female": 0,
                "Male": 1
            },

            "Exercise": {
                "No": 0,
                "Yes": 1
            },

            "Smoking": {
                "No": 0,
                "Yes": 1
            },

            "Academic_Year": {
                "1st": 1,
                "2nd": 2,
                "3rd": 3,
                "4th": 4
            },

            "Three_Level": {
                "Low": 1,
                "Medium": 2,
                "High": 3
            }
        }
    }

    with open(
        ARTIFACT_DIR / "feature_config.json",
        "w"
    ) as file:

        json.dump(
            feature_config,
            file,
            indent=2
        )

    print("\n==============================")
    print("FILES CREATED")
    print("==============================")

    print(
        ARTIFACT_DIR / "best_model.joblib"
    )

    print(
        ARTIFACT_DIR / "model_metrics.json"
    )

    print(
        ARTIFACT_DIR / "feature_config.json"
    )


if __name__ == "__main__":
    main()