from django.urls import path
from .views import water_analysis

urlpatterns = [
    path('', water_analysis, name='water_analysis'),
]
