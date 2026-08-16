"""Run Dash using local-by-default environment settings."""

from app.app import app
from src.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    app.run(debug=False, host=settings.host, port=settings.port)
