from django.db import models

# Create your models here.
class Image(models.Model):
    file = models.ImageField(upload_to='images/')
    caption = models.CharField(max_length=255, blank=True)