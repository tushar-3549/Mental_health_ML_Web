from rest_framework import serializers

FEATURES = [
    "Gender", "Age", "Academic Year", "Sleep Hours",
    "Social Media Usage", "Exercise", "Smoking",
    "Academic Pressure", "Academic Satisfaction", "Career Concern",
    "Financial Stress", "Family Pressure", "Feeling Depressed",
    "Feeling Anxious", "Feeling Stressed", "Difficulty Concentrating",
    "Feeling Lonely", "Feeling Isolated",
]

class PredictionSerializer(serializers.Serializer):
    Gender = serializers.CharField()
    Age = serializers.IntegerField(min_value=10, max_value=100)
    Academic_Year = serializers.CharField()
    Sleep_Hours = serializers.FloatField(min_value=0, max_value=24)
    Social_Media_Usage = serializers.CharField()
    Exercise = serializers.CharField()
    Smoking = serializers.CharField()
    Academic_Pressure = serializers.CharField()
    Academic_Satisfaction = serializers.CharField()
    Career_Concern = serializers.CharField()
    Financial_Stress = serializers.CharField()
    Family_Pressure = serializers.CharField()
    Feeling_Depressed = serializers.CharField()
    Feeling_Anxious = serializers.CharField()
    Feeling_Stressed = serializers.CharField()
    Difficulty_Concentrating = serializers.CharField()
    Feeling_Lonely = serializers.CharField()
    Feeling_Isolated = serializers.CharField()

    def to_internal_value(self, data):
        normalized = dict(data)
        mapping = {
            "Academic_Year": "Academic Year",
            "Sleep_Hours": "Sleep Hours",
            "Social_Media_Usage": "Social Media Usage",
            "Academic_Pressure": "Academic Pressure",
            "Academic_Satisfaction": "Academic Satisfaction",
            "Career_Concern": "Career Concern",
            "Financial_Stress": "Financial Stress",
            "Family_Pressure": "Family Pressure",
            "Feeling_Depressed": "Feeling Depressed",
            "Feeling_Anxious": "Feeling Anxious",
            "Feeling_Stressed": "Feeling Stressed",
            "Difficulty_Concentrating": "Difficulty Concentrating",
            "Feeling_Lonely": "Feeling Lonely",
            "Feeling_Isolated": "Feeling Isolated",
        }
        result = super().to_internal_value(normalized)
        return result
