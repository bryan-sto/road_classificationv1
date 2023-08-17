from django.db import models

class Prediction(models.Model):
    filename = models.CharField(max_length=255)
    prediction = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.filename} - {self.prediction}"
