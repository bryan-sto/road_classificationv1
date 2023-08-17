from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_images, name='predict_images'),
    path('predict-api/', views.predict_images_api, name='predict_images_api'),  # Add this line
]
