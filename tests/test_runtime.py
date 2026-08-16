"""Runtime-boundary and health-endpoint tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.settings import get_settings


class RuntimeTests(unittest.TestCase):
    def test_local_defaults_are_private_and_available(self) -> None:
        settings = get_settings()
        self.assertEqual(settings.runtime_mode, "local")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertTrue(settings.workbook_path.exists())
        self.assertTrue(settings.metrics_path.exists())
        self.assertTrue(settings.query_intents_path.exists())

    def test_health_and_readiness_endpoints(self) -> None:
        from app.app import app

        client = app.server.test_client()
        health = client.get("/healthz")
        readiness = client.get("/readyz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(readiness.get_json()["status"], "ready")

    def test_container_image_excludes_workbooks_and_models(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        dockerignore = (project_root / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("data/*.xlsx", dockerignore)
        self.assertIn("models/*.joblib", dockerignore)


if __name__ == "__main__":
    unittest.main()
