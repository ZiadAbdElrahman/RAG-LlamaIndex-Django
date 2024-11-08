from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='data/pdf/')
    upload_date = models.DateTimeField(auto_now_add=True)