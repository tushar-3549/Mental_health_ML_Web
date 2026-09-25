from django.urls import path
from .views import predict_api

urlpatterns = [
    path("predict/", predict_api, name="predict-api"),
]
