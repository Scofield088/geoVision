from django.urls import path
from .views import carbon_analysis

urlpatterns = [
    path("", carbon_analysis, name="carbon_analysis"),
]
