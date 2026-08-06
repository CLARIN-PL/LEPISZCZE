#!/usr/bin/env python3
"""Build and validate the static LEPISZCZE leaderboard data snapshot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
SOURCE_REPOSITORY = "https://github.com/CLARIN-PL/embeddings"
DEFAULT_SOURCE_COMMIT = "a434193a871c7d1bccdb5fd9106be5a950e38aad"

DATASET_LABELS = {
    "allegro/klej-cdsc-e": "CDSC-E",
    "allegro/klej-dyk": "DYK",
    "allegro/klej-polemo2-in": "PolEmo 2.0 In-Domain",
    "allegro/klej-polemo2-out": "PolEmo 2.0 Out-of-Domain",
    "allegro/klej-psc": "PSC",
    "clarin-pl/2021-punctuation-restoration": "Punctuation Restoration",
    "clarin-pl/aspectemo": "AspectEmo",
    "clarin-pl/dialogue-acts": "Dialogue Acts",
    "clarin-pl/kpwr-ner": "KPWr NER",
    "clarin-pl/nkjp-pos": "NKJP POS",
    "clarin-pl/polemo2-official": "PolEmo 2.0",
    "laugustyniak/abusive-clauses-pl": "Abusive Clauses",
    "laugustyniak/political-advertising-pl": "Political Advertising",
    "qa_all": "QA · Combined",
    "qa_kpwr": "QA · KPWr",
    "qa_nkjp": "QA · NKJP",
    "qa_wikipedia": "QA · Wikipedia",
}


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    return numeric


def _metric_names(raw: dict[str, Any]) -> Iterable[str]:
    for name, value in raw.get("metrics_avg", {}).items():
        if name != "classes" and isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name


def build_snapshot(source_dir: Path, source_commit: str) -> dict[str, Any]:
    """Convert historical result files into the public, minimal data schema."""
    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        raise ValueError(f"No JSON result files found in {source_dir}")

    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for source_file in source_files:
        raw = json.loads(source_file.read_text(encoding="utf-8"))
        entry_id = str(raw["submission_name"])
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate submission_name: {entry_id}")
        seen_ids.add(entry_id)

        dataset = str(raw["dataset_name"])
        task = str(raw["leaderboard_task_name"])
        averages = raw.get("metrics_avg", {})
        deviations = raw.get("metrics_std", {})
        metrics: dict[str, dict[str, float | None]] = {}

        for metric_name in sorted(_metric_names(raw)):
            mean = _finite_number(averages[metric_name], f"{entry_id}.{metric_name}.mean")
            std_value = deviations.get(metric_name)
            std = (
                _finite_number(std_value, f"{entry_id}.{metric_name}.std")
                if std_value is not None
                else None
            )
            metrics[metric_name] = {"mean": mean, "std": std}

        if not metrics:
            raise ValueError(f"No summary metrics found for {entry_id}")

        entries.append(
            {
                "id": entry_id,
                "task": task,
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model": str(raw["embedding_name"]),
                "dataset_version": str(raw.get("dataset_version", "unknown")),
                "averaged_over": raw.get("averaged_over"),
                "metrics": metrics,
            }
        )

    entries.sort(
        key=lambda item: (
            item["task"] == "Question Answering",
            item["task"],
            item["dataset"],
            item["model"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": source_commit,
            "results_path": "webpage/data/results",
            "license": "MIT",
        },
        "summary": {
            "entries": len(entries),
            "datasets": len({entry["dataset"] for entry in entries}),
            "models": len({entry["model"] for entry in entries}),
            "tasks": len({entry["task"] for entry in entries}),
        },
        "entries": entries,
    }


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail fast when the checked-in public snapshot is malformed."""
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {snapshot.get('schema_version')}")

    source = snapshot.get("source")
    if not isinstance(source, dict) or not source.get("repository") or not source.get("commit"):
        raise ValueError("Snapshot source metadata is incomplete")

    entries = snapshot.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Snapshot must contain at least one result")

    seen_ids: set[str] = set()
    required = {"id", "task", "dataset", "dataset_label", "model", "metrics"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entries[{index}] must be an object")
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"entries[{index}] is missing: {', '.join(sorted(missing))}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate result id: {entry['id']}")
        seen_ids.add(entry["id"])
        if not isinstance(entry["metrics"], dict) or not entry["metrics"]:
            raise ValueError(f"No metrics for {entry['id']}")
        for metric_name, values in entry["metrics"].items():
            if not isinstance(values, dict) or "mean" not in values:
                raise ValueError(f"Invalid metric {entry['id']}.{metric_name}")
            _finite_number(values["mean"], f"{entry['id']}.{metric_name}.mean")
            if values.get("std") is not None:
                _finite_number(values["std"], f"{entry['id']}.{metric_name}.std")

    summary = snapshot.get("summary", {})
    expected_summary = {
        "entries": len(entries),
        "datasets": len({entry["dataset"] for entry in entries}),
        "models": len({entry["model"] for entry in entries}),
        "tasks": len({entry["task"] for entry in entries}),
    }
    if summary != expected_summary:
        raise ValueError(f"Summary mismatch: expected {expected_summary}, got {summary}")


def write_snapshot(snapshot: dict[str, Any], output: Path) -> None:
    """Write stable JSON output so rebuilds produce reviewable diffs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-dir", type=Path, help="Directory containing historical result JSON files")
    mode.add_argument("--check", type=Path, help="Validate an existing public snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data/leaderboard.json"),
        help="Output path used with --source-dir",
    )
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        snapshot = json.loads(args.check.read_text(encoding="utf-8"))
        validate_snapshot(snapshot)
        print(f"Validated {len(snapshot['entries'])} leaderboard results in {args.check}")
        return

    snapshot = build_snapshot(args.source_dir, args.source_commit)
    validate_snapshot(snapshot)
    write_snapshot(snapshot, args.output)
    print(f"Wrote {len(snapshot['entries'])} leaderboard results to {args.output}")


if __name__ == "__main__":
    main()
