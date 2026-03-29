"""
polyguard/asgi.py
=================
ASGI entry point consumed by uvicorn / gunicorn.

    uvicorn polyguard.asgi:app --reload --port 8000
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker polyguard.asgi:app
"""

from api.app import create_app

app = create_app()