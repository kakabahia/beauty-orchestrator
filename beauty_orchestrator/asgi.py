"""ASGI config for beauty_orchestrator project."""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "beauty_orchestrator.settings")
application = get_asgi_application()
