# from django.shortcuts import render
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status

# from .serializers import PredictionSerializer
# from .ml_service import get_ml_service
# from .recommendation import build_recommendations
# from .models import PredictionHistory

# def home(request):
#     return render(request, "home.html")

# def predict_page(request):
#     return render(request, "prediction.html")

# def comparison_page(request):
#     try:
#         metrics = get_ml_service().metrics()
#     except Exception:
#         metrics = []
#     return render(request, "comparison.html", {"metrics": metrics})

# @api_view(["POST"])
# def predict_api(request):
#     serializer = PredictionSerializer(data=request.data)
#     serializer.is_valid(raise_exception=True)

#     clean = serializer.validated_data

#     # Convert serializer keys to dataset feature names.
#     data = {
#         "Gender": clean["Gender"],
#         "Age": clean["Age"],
#         "Academic Year": clean["Academic_Year"],
#         "Sleep Hours": clean["Sleep_Hours"],
#         "Social Media Usage": clean["Social_Media_Usage"],
#         "Exercise": clean["Exercise"],
#         "Smoking": clean["Smoking"],
#         "Academic Pressure": clean["Academic_Pressure"],
#         "Academic Satisfaction": clean["Academic_Satisfaction"],
#         "Career Concern": clean["Career_Concern"],
#         "Financial Stress": clean["Financial_Stress"],
#         "Family Pressure": clean["Family_Pressure"],
#         "Feeling Depressed": clean["Feeling_Depressed"],
#         "Feeling Anxious": clean["Feeling_Anxious"],
#         "Feeling Stressed": clean["Feeling_Stressed"],
#         "Difficulty Concentrating": clean["Difficulty_Concentrating"],
#         "Feeling Lonely": clean["Feeling_Lonely"],
#         "Feeling Isolated": clean["Feeling_Isolated"],
#     }

#     try:
#         service = get_ml_service()
#         prediction, confidence, probabilities = service.predict(data)
#     except FileNotFoundError as exc:
#         return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

#     recommendations = build_recommendations(data, prediction)

#     PredictionHistory.objects.create(
#         risk_level=prediction,
#         confidence=confidence,
#         input_data=data,
#         recommendations=recommendations,
#     )

#     return Response({
#         "prediction": prediction,
#         "confidence": round(confidence, 2) if confidence is not None else None,
#         "recommendations": recommendations,
#     })


from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import PredictionSerializer
from .ml_service import get_ml_service
from .recommendation import build_recommendations
from .models import PredictionHistory


def home(request):
    return render(request, "home.html")


def predict_page(request):
    return render(request, "prediction.html")


# def comparison_page(request):
#     try:
#         metrics = get_ml_service().metrics()
#     except Exception:
#         metrics = []

#     return render(request, "comparison.html", {"metrics": metrics})


def comparison_page(request):
    try:
        raw_metrics = get_ml_service().metrics()

        metrics = []

        for item in raw_metrics:
            metrics.append({
                "model": item.get("Model", item.get("model", "")),
                "accuracy": item.get("Accuracy", item.get("accuracy", 0)),
                "precision": item.get("Precision", item.get("precision", 0)),
                "recall": item.get("Recall", item.get("recall", 0)),
                "f1_score": item.get(
                    "F1 Score",
                    item.get("f1_score", 0)
                ),
            })

    except Exception as exc:
        print("Comparison error:", exc)
        metrics = []

    return render(
        request,
        "comparison.html",
        {"metrics": metrics}
    )


@api_view(["POST"])
def predict_api(request):

    serializer = PredictionSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            {
                "detail": "Invalid input data.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    clean = serializer.validated_data

    # IMPORTANT:
    # These names must match the columns used during ML training.
    data = {
        "Gender": clean["Gender"],
        "Age": clean["Age"],
        "Academic_Year": clean["Academic_Year"],
        "Sleep_Hours": clean["Sleep_Hours"],
        "Social_Media_Usage": clean["Social_Media_Usage"],
        "Exercise": clean["Exercise"],
        "Smoking": clean["Smoking"],
        "Academic_Pressure": clean["Academic_Pressure"],
        "Academic_Satisfaction": clean["Academic_Satisfaction"],
        "Career_Concern": clean["Career_Concern"],
        "Financial_Stress": clean["Financial_Stress"],
        "Family_Pressure": clean["Family_Pressure"],
        "Feeling_Depressed": clean["Feeling_Depressed"],
        "Feeling_Anxious": clean["Feeling_Anxious"],
        "Feeling_Stressed": clean["Feeling_Stressed"],
        "Difficulty_Concentrating": clean["Difficulty_Concentrating"],
        "Feeling_Lonely": clean["Feeling_Lonely"],
        "Feeling_Isolated": clean["Feeling_Isolated"],
    }

    try:
        service = get_ml_service()

        prediction, confidence, probabilities = service.predict(data)

    except FileNotFoundError as exc:
        return Response(
            {"detail": str(exc)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as exc:
        # Return the actual ML error as JSON instead of Django HTML 500 page.
        return Response(
            {
                "detail": "ML prediction failed.",
                "error": str(exc),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Convert numeric prediction to human-readable label
    prediction_labels = {
        0: "Normal",
        1: "Moderate Risk",
        2: "High Risk",
    }

    try:
        prediction_value = int(prediction)
        prediction_label = prediction_labels.get(
            prediction_value,
            str(prediction),
        )
    except (ValueError, TypeError):
        prediction_label = str(prediction)

    # Build recommendations
    recommendations = build_recommendations(
        data,
        prediction_label,
    )

    # Save prediction history
    PredictionHistory.objects.create(
        risk_level=prediction_label,
        confidence=confidence,
        input_data=data,
        recommendations=recommendations,
    )

    return Response(
        {
            "prediction": prediction_label,
            "confidence": (
                round(confidence, 2)
                if confidence is not None
                else None
            ),
            "recommendations": recommendations,
        },
        status=status.HTTP_200_OK,
    )