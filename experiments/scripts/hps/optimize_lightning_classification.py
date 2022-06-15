import typer
from embeddings.config.lighting_config_space import LightingTextClassificationConfigSpace
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingClassificationPipeline
from embeddings.utils.utils import build_output_path

from lepiszcze.logging import get_hps_run_name
from lepiszcze.paths import (
    HPS_COMMON_CFG_PATH,
    LIGHTNING_HPS_OUTPUT_PATH,
    get_dataset_config_path,
    get_embedding_name_by_path,
)
from lepiszcze.setup import disable_hf_datasets_caching, get_lightning_logging_config
from lepiszcze.utils import parse_dataset_cfg_for_evaluation, read_yaml

app = typer.Typer()


def run(
    embedding_path: str = typer.Option("...", help="Embedding path."),
    ds: str = typer.Option("...", help="Dataset name."),
    hps_config_path: str = typer.Option("...", help="HPS config path."),
) -> None:
    disable_hf_datasets_caching()
    config = read_yaml(hps_config_path)

    early_stopping_kwargs = config["early_stopping_kwargs"]
    hps_config = config["classification"]
    hps_config["embedding_name_or_path"] = get_embedding_name_by_path(embedding_path)
    hps_config_common = read_yaml(HPS_COMMON_CFG_PATH)
    config_space = LightingTextClassificationConfigSpace.from_dict(hps_config)

    (
        dataset_name,
        dataset_path,
        input_column_name,
        target_column_name,
    ) = parse_dataset_cfg_for_evaluation(str(get_dataset_config_path(ds)), cfg_type="lightning_hps")
    output_path = build_output_path(
        LIGHTNING_HPS_OUTPUT_PATH, embedding_path, dataset_name, timestamp_subdir=False, mkdirs=True
    )
    logging_config = get_lightning_logging_config(dataset_name)
    optimized_pipeline = OptimizedLightingClassificationPipeline(
        config_space=config_space,  # type: ignore
        dataset_name_or_path=dataset_path,
        input_column_name=input_column_name,  # type: ignore
        target_column_name=target_column_name,
        ignore_preprocessing_pipeline=True,
        logging_config=logging_config,
        early_stopping_kwargs=early_stopping_kwargs,
        **hps_config_common,
    ).persisting(
        best_params_path=output_path / "best_params.yaml",
        log_path=output_path / "hps_log.pickle",
    )
    optimized_pipeline.run(
        run_name=get_hps_run_name(dataset=dataset_name, embedding_path=embedding_path)
    )


typer.run(run)
