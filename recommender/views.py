from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index_view(request: HttpRequest) -> HttpResponse:
    """Renders the Adaptive Dual-Flow Interface (Tailwind CSS + Alpine.js via CDN)."""
    return render(request, "index.html")
