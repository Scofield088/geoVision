from django.urls import path
from .views import urban_analysis

urlpatterns = [
    path("", urban_analysis, name="urban_analysis"),
]
