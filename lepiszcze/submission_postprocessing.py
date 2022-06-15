import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict


@dataclass
class ValueExtractor:
    metric_name: str
    decimal: int = 4

    def extract_value(self, raw_data: Dict[str, Any]) -> Tuple[float, float]:
        return round(raw_data["metrics_avg"][self.metric_name], self.decimal), round(
            raw_data["metrics_std"][self.metric_name], self.decimal
        )


class SubmissionsDetails(TypedDict):
    dataset_name: str
    model_name: str
    metrics_tuple: Tuple[float, float]


def parse_json_file(json_file: Path, value_extractor: ValueExtractor) -> SubmissionsDetails:
    submission_raw = json.load(open(json_file, "r"))
    parsed: SubmissionsDetails = dict(
        dataset_name=Path(submission_raw["dataset_name"]["value"]).parent.name,
        model_name=submission_raw["embedding_name"]["value"],
        metrics_tuple=value_extractor.extract_value(submission_raw),
    )
    return parsed
