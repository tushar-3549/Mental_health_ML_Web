from django.contrib import admin
from django.urls import path, include
from prediction import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("predict/", views.predict_page, name="predict"),
    path("comparison/", views.comparison_page, name="comparison"),
    path("api/", include("prediction.urls")),
]
