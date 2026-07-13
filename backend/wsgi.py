"""Deployment entry point for WSGI servers such as Waitress or Gunicorn."""

from backend.app import app


application = app
