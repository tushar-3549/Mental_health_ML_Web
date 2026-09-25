from django.db import models

class PredictionHistory(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    risk_level = models.CharField(max_length=50)
    confidence = models.FloatField(null=True, blank=True)
    input_data = models.JSONField(default=dict)
    recommendations = models.JSONField(default=list)

    def __str__(self):
        return f"{self.risk_level} - {self.created_at}"
