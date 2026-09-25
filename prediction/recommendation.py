def _level(value):
    return str(value).strip().lower()

def build_recommendations(data, risk_level):
    recs = []

    if float(data.get("Sleep Hours", 0)) < 7:
        recs.append("Improve sleep habits and aim for a consistent sleep schedule.")

    if _level(data.get("Exercise", "")) in {"no", "none"}:
        recs.append("Include regular physical activity in your weekly routine.")

    if _level(data.get("Social Media Usage", "")) in {"high", "very high"}:
        recs.append("Consider reducing excessive social media usage and screen time.")

    stress_fields = [
        "Academic Pressure", "Financial Stress", "Family Pressure",
        "Feeling Stressed", "Feeling Anxious"
    ]
    if any(_level(data.get(x, "")) in {"high", "very high"} for x in stress_fields):
        recs.append("Practice relaxation, breathing, mindfulness, or meditation techniques.")

    if any(_level(data.get(x, "")) in {"high", "very high"} for x in [
        "Feeling Depressed", "Feeling Anxious", "Feeling Lonely", "Feeling Isolated"
    ]):
        recs.append("Consider talking with a counselor, trusted person, or qualified mental-health professional.")

    if _level(data.get("Smoking", "")) in {"yes", "frequent"}:
        recs.append("Consider reducing or stopping smoking and seek professional support if needed.")

    if not recs:
        recs.append("Maintain your current healthy routines and continue monitoring your wellbeing.")

    if str(risk_level).strip().lower() == "high risk":
        recs.append("Because the predicted risk is high, consider seeking support from a qualified professional.")

    return list(dict.fromkeys(recs))
