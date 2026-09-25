from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ("risk_level", "confidence", "created_at")
    list_filter = ("risk_level", "created_at")
    readonly_fields = ("created_at",)
