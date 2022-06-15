import importlib
import json
import os.path
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Type, Union

import yaml
from pygments import highlight  # type: ignore
from pygments.formatters.terminal256 import Terminal256Formatter  # type: ignore
from pygments.lexers.data import JsonLexer  # type: ignore
from yaml.loader import Loader


def read_yaml(filepath: Union[str, Path], safe_load: bool = True) -> Any:
    with open(filepath, "r") as f:
        if safe_load:
            return yaml.safe_load(f)
        else:
            return yaml.load(f, Loader=Loader)


def get_module_from_str(module: str) -> Type[Any]:
    module, cls = module.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module), cls)
    return cls  # type: ignore


def get_create_eval_paths(
    results_path: Path, models_path: Path, dataset_name: str, embedding_name: str
) -> Tuple[str, str]:
    persist_out_path = results_path.joinpath(dataset_name, f"{embedding_name}.json")
    persist_out_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = models_path.joinpath(dataset_name, embedding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(persist_out_path), str(output_path)


def prepare_output_path(path: str) -> Path:
    output_dir = Path(os.path.dirname(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(path)


def pprint_dict(dictionary: Dict[str, float]) -> None:
    raw_json = json.dumps(dictionary, indent=2)
    colorful = highlight(
        raw_json,
        lexer=JsonLexer(),
        formatter=Terminal256Formatter(),
    )
    print(colorful)


def parse_dataset_cfg_for_evaluation(
    dataset_cfg_path: str, cfg_type: str
) -> Tuple[str, str, Union[str, Sequence[str]], str]:
    ds_cfg = read_yaml(dataset_cfg_path)
    return (
        ds_cfg["name"],
        ds_cfg["paths"][cfg_type],
        ds_cfg["common_args"]["input_column_names"],
        ds_cfg["common_args"]["target_column_name"],
    )
