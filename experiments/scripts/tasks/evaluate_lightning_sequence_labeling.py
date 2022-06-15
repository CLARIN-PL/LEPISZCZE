from copy import deepcopy

import typer
import wandb
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.utils.utils import build_output_path
from tqdm.auto import tqdm

from lepiszcze.defaults import get_model_checkpoint_kwargs
from lepiszcze.logging import get_task_run_name
from lepiszcze.paths import (
    LIGHTNING_PIPELINE_OUTPUT_PATH,
    get_dataset_config_path,
    get_lightning_optimized_pipeline_params_path,
)
from lepiszcze.setup import disable_hf_datasets_caching, get_lightning_logging_config
from lepiszcze.utils import parse_dataset_cfg_for_evaluation, read_yaml

app = typer.Typer()


def run(
    embedding_path: str = typer.Option("...", help="Embedding path."),
    ds: str = typer.Option("...", help="Dataset name."),
    retrains: int = typer.Option("...", help="Number of model retrains."),
) -> None:
    disable_hf_datasets_caching()

    (
        dataset_name,
        dataset_path,
        input_column_name,
        target_column_name,
    ) = parse_dataset_cfg_for_evaluation(str(get_dataset_config_path(ds)), cfg_type="lightning")
    cfg = read_yaml(
        get_lightning_optimized_pipeline_params_path(
            embedding_path=embedding_path, dataset_name=ds
        ),
        safe_load=False,
    )
    output_path = build_output_path(
        LIGHTNING_PIPELINE_OUTPUT_PATH,
        embedding_path,
        dataset_name,
        timestamp_subdir=False,
        mkdirs=True,
    )

    cfg["dataset_name_or_path"] = dataset_path
    cfg["output_path"] = output_path
    cfg["logging_config"] = get_lightning_logging_config(dataset_name)
    cfg["model_checkpoint_kwargs"] = get_model_checkpoint_kwargs()

    for run_id in tqdm(range(retrains), desc="Run"):
        run_cfg = deepcopy(cfg)
        run_output_path = output_path / f"run-{run_id}"
        run_output_path.mkdir(parents=False, exist_ok=False)
        run_cfg["output_path"] = run_output_path
        pipeline = LightningSequenceLabelingPipeline(**run_cfg)
        run_name = get_task_run_name(
            dataset=dataset_name, embedding_path=embedding_path, run_id=run_id
        )
        pipeline.run(run_name=run_name)
        wandb.restore(
            name="config.yaml",
            run_path=pipeline.model.task.trainer.logger[0]._experiment.path,  # type: ignore
            root=str(run_output_path),
        )


typer.run(run)
