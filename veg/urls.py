from django.urls import path
from .views import veg_analysis

urlpatterns = [
    path('', veg_analysis, name='veg_analysis'),
]
