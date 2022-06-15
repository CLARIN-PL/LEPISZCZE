from typing import Type, Union

import typer
from datasets import DatasetDict
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from flair.data import Corpus

from lepiszcze.paths import get_dataset_config_path
from lepiszcze.setup import disable_hf_datasets_caching
from lepiszcze.utils import get_module_from_str, prepare_output_path, read_yaml

app = typer.Typer()


def run(
    ds: str = typer.Option(..., help="Dataset name."),
    cfg_type: str = typer.Option(..., help="Config type."),
    is_hps: bool = typer.Option(False, help="Is HPS Dataset"),
) -> None:
    cfg_type = cfg_type if not is_hps else f"{cfg_type}_hps"
    disable_hf_datasets_caching()
    cfg = read_yaml(get_dataset_config_path(ds))["datasets"][cfg_type]

    pipeline_kwargs = cfg["pipeline_args"]
    pipeline_kwargs["persist_path"] = prepare_output_path(cfg["output"])
    pipeline_cls: Type[
        Union[
            PreprocessingPipeline[str, DatasetDict, Corpus],
            PreprocessingPipeline[str, DatasetDict, DatasetDict],
        ]
    ] = get_module_from_str(cfg["pipeline_cls"])
    pipeline = pipeline_cls(**pipeline_kwargs)
    pipeline.run()


typer.run(run)
