# Student Mental Health Risk Prediction Web App

Thesis project:
**A Comparative Analysis of ML Algorithms for Students Mental Health Risk Prediction**

## Stack
- Python
- Django
- Django REST Framework
- Scikit-learn
- Pandas
- Bootstrap 5
- SQLite (development)

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put your dataset here:

```text
dataset/mental_health.csv
```

The target column must be:

```text
Mental Health Condition
```

Expected feature names:

- Gender
- Age
- Academic Year
- Sleep Hours
- Social Media Usage
- Exercise
- Smoking
- Academic Pressure
- Academic Satisfaction
- Career Concern
- Financial Stress
- Family Pressure
- Feeling Depressed
- Feeling Anxious
- Feeling Stressed
- Difficulty Concentrating
- Feeling Lonely
- Feeling Isolated

## 2. Train and compare models

```bash
python ml_training/train.py
```

The script:
- loads the CSV
- cleans column names
- removes empty rows
- creates preprocessing pipeline
- compares Logistic Regression, Random Forest, SVM and Gradient Boosting
- calculates Accuracy, Precision, Recall and F1
- saves every model
- selects the best model by macro F1
- saves comparison metrics and feature configuration

Artifacts are saved in `ml_artifacts/`.

## 3. Run Django

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

API:

```text
POST /api/predict/
GET  /api/model-comparison/
```

## Important
This application is a research/demo risk-prediction system, not a medical diagnosis or emergency assessment tool.
