#!/usr/bin/env python3
"""Validate the public LEPISZCZE dataset registry."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 1
PROJECTS = {"AITAX", "JuDDGES"}
ARTIFACT_TYPES = {
    "benchmark_dataset",
    "enriched_corpus",
    "graph_dataset",
    "instruction_dataset",
    "source_corpus",
}
MATURITY_LEVELS = {"benchmark_candidate", "public_dataset", "public_preview"}
REQUIRED_TEXT_FIELDS = {
    "id",
    "name",
    "project",
    "artifact_type",
    "maturity",
    "language",
    "scope",
    "summary",
    "size_label",
}


def require_text(item: dict[str, Any], field: str, context: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value


def validate_registry(registry: dict[str, Any]) -> None:
    """Fail when registry data cannot be rendered safely and consistently."""
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {registry.get('schema_version')}")

    updated_at = registry.get("updated_at")
    if not isinstance(updated_at, str):
        raise ValueError("updated_at must be an ISO date")
    date.fromisoformat(updated_at)

    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")

    seen_ids: set[str] = set()
    for index, item in enumerate(datasets):
        context = f"datasets[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{context} must be an object")
        for field in REQUIRED_TEXT_FIELDS:
            require_text(item, field, context)

        dataset_id = str(item["id"])
        if dataset_id in seen_ids:
            raise ValueError(f"Duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)

        if item["project"] not in PROJECTS:
            raise ValueError(f"Unsupported project for {dataset_id}: {item['project']}")
        if item["artifact_type"] not in ARTIFACT_TYPES:
            raise ValueError(f"Unsupported artifact_type for {dataset_id}: {item['artifact_type']}")
        if item["maturity"] not in MATURITY_LEVELS:
            raise ValueError(f"Unsupported maturity for {dataset_id}: {item['maturity']}")
        if not isinstance(item.get("featured"), bool):
            raise ValueError(f"{context}.featured must be boolean")

        validation = item.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"{context}.validation must be an object")
        require_text(validation, "human_validation", f"{context}.validation")
        if not isinstance(validation.get("leaderboard_eligible"), bool):
            raise ValueError(f"{context}.validation.leaderboard_eligible must be boolean")
        missing = validation.get("missing")
        if not isinstance(missing, list) or not all(
            isinstance(value, str) and value.strip() for value in missing
        ):
            raise ValueError(f"{context}.validation.missing must be a list of strings")
        if validation["leaderboard_eligible"] and missing:
            raise ValueError(f"Leaderboard-eligible dataset {dataset_id} cannot have missing gates")

        links = item.get("links")
        if not isinstance(links, dict) or "dataset" not in links:
            raise ValueError(f"{context}.links must include dataset")
        for label, url in links.items():
            if not isinstance(label, str) or not isinstance(url, str):
                raise ValueError(f"{context}.links must map strings to strings")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Invalid HTTPS URL for {dataset_id}.{label}: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=Path("docs/data/datasets.json"),
    )
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_registry(registry)
    print(f"Validated {len(registry['datasets'])} datasets in {args.registry}")


if __name__ == "__main__":
    main()
