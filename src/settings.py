"""Environment-driven runtime settings for local and container execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_setting(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class RuntimeSettings:
    runtime_mode: str
    host: str
    port: int
    workbook_path: Path
    metrics_path: Path
    query_intents_path: Path
    model_dir: Path

    @property
    def is_local_only(self) -> bool:
        return self.runtime_mode in {"local", "local-container"}

    @property
    def boundary_label(self) -> str:
        return "LOCAL ONLY" if self.is_local_only else "PRIVATE RUNTIME"

    @property
    def boundary_message(self) -> str:
        if self.is_local_only:
            return "No data was sent outside this machine."
        return "Data remained inside the configured private application runtime."


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """Read runtime settings once per process."""

    return RuntimeSettings(
        runtime_mode=os.environ.get("SALES_AI_RUNTIME_MODE", "local").strip().lower(),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        workbook_path=_path_setting(
            "SALES_AI_WORKBOOK_PATH",
            PROJECT_ROOT / "data" / "MSP_Sales_Performance_Raw_Data_With_Common_Metrics.xlsx",
        ),
        metrics_path=_path_setting(
            "SALES_AI_METRICS_PATH", PROJECT_ROOT / "config" / "sales_metrics.yaml"
        ),
        query_intents_path=_path_setting(
            "SALES_AI_QUERY_INTENTS_PATH", PROJECT_ROOT / "config" / "query_intents.yaml"
        ),
        model_dir=_path_setting("SALES_AI_MODEL_DIR", PROJECT_ROOT / "models"),
    )

