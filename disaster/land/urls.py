from django.urls import path
from .views import land_anal

urlpatterns = [
    path("", land_anal, name="land_analysis"),
]
