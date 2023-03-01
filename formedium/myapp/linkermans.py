from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.index, name='index'),
    path('developer-portfolio',views.dev, name='dev'),
    path('3d', views.renders, name='3d'),
    path('upload', views.upload_image, name='upload') 
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
