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

def comparison_page(request):
    try:
        metrics = get_ml_service().metrics()
    except Exception:
        metrics = []
    return render(request, "comparison.html", {"metrics": metrics})

@api_view(["POST"])
def predict_api(request):
    serializer = PredictionSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    clean = serializer.validated_data

    # Convert serializer keys to dataset feature names.
    data = {
        "Gender": clean["Gender"],
        "Age": clean["Age"],
        "Academic Year": clean["Academic_Year"],
        "Sleep Hours": clean["Sleep_Hours"],
        "Social Media Usage": clean["Social_Media_Usage"],
        "Exercise": clean["Exercise"],
        "Smoking": clean["Smoking"],
        "Academic Pressure": clean["Academic_Pressure"],
        "Academic Satisfaction": clean["Academic_Satisfaction"],
        "Career Concern": clean["Career_Concern"],
        "Financial Stress": clean["Financial_Stress"],
        "Family Pressure": clean["Family_Pressure"],
        "Feeling Depressed": clean["Feeling_Depressed"],
        "Feeling Anxious": clean["Feeling_Anxious"],
        "Feeling Stressed": clean["Feeling_Stressed"],
        "Difficulty Concentrating": clean["Difficulty_Concentrating"],
        "Feeling Lonely": clean["Feeling_Lonely"],
        "Feeling Isolated": clean["Feeling_Isolated"],
    }

    try:
        service = get_ml_service()
        prediction, confidence, probabilities = service.predict(data)
    except FileNotFoundError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    recommendations = build_recommendations(data, prediction)

    PredictionHistory.objects.create(
        risk_level=prediction,
        confidence=confidence,
        input_data=data,
        recommendations=recommendations,
    )

    return Response({
        "prediction": prediction,
        "confidence": round(confidence, 2) if confidence is not None else None,
        "recommendations": recommendations,
    })
