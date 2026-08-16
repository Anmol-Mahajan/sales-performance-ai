"""Main Dash app with Sales Manager and Model Analysis portals."""

from __future__ import annotations

from dash import Dash, dcc, html

from src.data_loader import load_sales_data
from src.settings import RuntimeSettings, get_settings

from .data_scientist_portal import build_data_scientist_layout
from .manager_portal import build_manager_layout, register_manager_callbacks


def _runtime_notice(settings: RuntimeSettings) -> str:
    if settings.is_local_only:
        return (
            "This application does not use the internet, external APIs, cloud models, hosted "
            "LLMs, or online data sources. The organisation-specific model is trained only "
            "from the local Excel workbook and saved locally."
        )
    return (
        "This private container runtime reads only its configured workbook and model storage. "
        "It does not call external APIs or hosted models. Deployment access, storage, identity, "
        "networking, retention, and residency must be governed by the hosting organisation."
    )


def create_app(runtime_settings: RuntimeSettings | None = None) -> Dash:
    """Create the Dash app for a local or configured private runtime."""

    settings = runtime_settings or get_settings()
    data = load_sales_data(settings.workbook_path, metrics_path=settings.metrics_path)
    app = Dash(
        __name__,
        title="Sales Performance AI",
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("Sales Performance AI", className="app-title"),
                            html.Div(
                                "Sales and customer performance intelligence using configured private data.",
                                className="app-subtitle",
                            ),
                        ]
                    ),
                    html.Div(settings.boundary_label, className="local-pill"),
                ],
                className="app-header",
            ),
            html.Div(_runtime_notice(settings), className="local-notice"),
            dcc.Tabs(
                id="portal-tabs",
                value="manager",
                children=[
                    dcc.Tab(
                        label="Sales Manager Portal",
                        value="manager",
                        children=build_manager_layout(data),
                    ),
                    dcc.Tab(
                        label="Model Analysis",
                        value="data-scientist",
                        children=build_data_scientist_layout(data),
                    ),
                ],
                className="tabs",
            ),
        ],
        className="app-shell",
    )
    register_manager_callbacks(app, data, settings.boundary_message)

    @app.server.get("/healthz")
    def health_check():
        return {"status": "ok", "runtime_mode": settings.runtime_mode}, 200

    @app.server.get("/readyz")
    def readiness_check():
        ready = settings.workbook_path.exists() and settings.metrics_path.exists()
        return {
            "status": "ready" if ready else "not_ready",
            "workbook_configured": settings.workbook_path.exists(),
            "metrics_configured": settings.metrics_path.exists(),
        }, 200 if ready else 503
    return app


app = create_app()
server = app.server
