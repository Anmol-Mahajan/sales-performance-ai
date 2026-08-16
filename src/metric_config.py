"""Load and apply canonical metric semantics from the local YAML file."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .settings import get_settings


DEFAULT_METRICS_PATH = get_settings().metrics_path


def load_metric_config(path: str | Path = DEFAULT_METRICS_PATH) -> dict:
    """Load metric definitions from a local YAML file with no network access."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Local metric configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if "available_derived_metrics" not in config:
        raise ValueError("Metric configuration is missing available_derived_metrics")
    return config


def _evaluate_formula(node: ast.AST, frame: pd.DataFrame):
    if isinstance(node, ast.Expression):
        return _evaluate_formula(node.body, frame)
    if isinstance(node, ast.Name):
        if node.id not in frame.columns:
            raise KeyError(node.id)
        return pd.to_numeric(frame[node.id], errors="coerce").fillna(0)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_formula(node.operand, frame)
    if isinstance(node, ast.BinOp):
        left = _evaluate_formula(node.left, frame)
        right = _evaluate_formula(node.right, frame)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if np.isscalar(right):
                return left / right if right else 0.0
            return left.div(right.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    raise ValueError(f"Unsupported metric formula element: {ast.dump(node)}")


def recompute_derived_metrics(
    monthly: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute YAML-supported metrics and return an execution report."""

    metric_config = config or load_metric_config()
    enriched = monthly.copy()
    report = []
    for metric, definition in metric_config.get("available_derived_metrics", {}).items():
        formula = str(definition.get("formula", "")).strip()
        try:
            tree = ast.parse(formula, mode="eval")
            enriched[metric] = _evaluate_formula(tree, enriched)
            report.append(
                {"Metric": metric, "Formula": formula, "Status": "Recomputed", "Message": "Calculated from local source fields"}
            )
        except (KeyError, ValueError, SyntaxError) as exc:
            report.append(
                {"Metric": metric, "Formula": formula, "Status": "Unavailable", "Message": str(exc)}
            )
    return enriched, pd.DataFrame(report)


def metric_catalogue(config: dict | None = None) -> pd.DataFrame:
    """Return YAML metric availability and governance as a displayable table."""

    metric_config = config or load_metric_config()
    groups = {
        metric: group
        for group, metrics in metric_config.get("metric_groups", {}).items()
        for metric in metrics
    }
    available = metric_config.get("available_derived_metrics", {})
    future = metric_config.get("requires_future_fields", {})
    names = list(dict.fromkeys([*groups, *available, *future]))
    rows = []
    for name in names:
        definition = available.get(name, {})
        future_definition = future.get(name, {})
        rows.append(
            {
                "Metric": name,
                "Group": groups.get(name, "other").replace("_", " ").title(),
                "Availability": "Available" if name in available else "Requires future fields" if name in future else "Source metric",
                "Formula": definition.get("formula", "Source field or event aggregation"),
                "Required Fields": ", ".join(future_definition.get("required_fields", [])),
            }
        )
    return pd.DataFrame(rows)
