"""URL configuration for beauty_orchestrator project."""
from django.contrib import admin
from django.urls import path
from recommender.api import api
from recommender.views import index_view

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api.urls),
    path("", index_view, name="index"),
]
