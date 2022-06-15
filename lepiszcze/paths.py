import os
from pathlib import Path

from . import ModelExecutionLibrary

ROOT_DIR = Path(os.path.dirname(__file__)).parent.absolute()

DATA_PATH = ROOT_DIR / "data"
DATASETS_PATH = DATA_PATH / "datasets"

# CONFIGS
EXPERIMENTS_PATH = ROOT_DIR / "experiments"
CONFIGS_PATH = EXPERIMENTS_PATH / "configs"
DATASETS_CFG_PATH = CONFIGS_PATH / "datasets"
HPS_COMMON_CFG_PATH = CONFIGS_PATH / "hps" / "common.yaml"

# OUTPUT PATHS
# HPS
HPS_OUTPUT_PATH = DATA_PATH / "hps"
LIGHTNING_HPS_OUTPUT_PATH = HPS_OUTPUT_PATH / "lightning"
FLAIR_HPS_OUTPUT_PATH = HPS_OUTPUT_PATH / "flair"

# PIPELINES
MODELS_OUTPUT_PATH = DATA_PATH / "models"
LIGHTNING_PIPELINE_OUTPUT_PATH = MODELS_OUTPUT_PATH / "lightning"
FLAIR_PIPELINE_OUTPUT_PATH = MODELS_OUTPUT_PATH / "flair"
PIPELINES_OUTPUT_PATHS_MAPPING = {
    "flair": FLAIR_PIPELINE_OUTPUT_PATH,
    "lightning": LIGHTNING_PIPELINE_OUTPUT_PATH,
}

# SUBMISSIONS
SUBMISSIONS_PATH = DATA_PATH / "submissions"
TABLES_PATH = DATA_PATH / "tables"


def get_dataset_config_path(dataset_name: str) -> Path:
    return DATASETS_CFG_PATH / f"{dataset_name}.yaml"


def get_lightning_optimized_pipeline_params_path(embedding_path: str, dataset_name: str) -> Path:
    return LIGHTNING_HPS_OUTPUT_PATH / embedding_path / dataset_name / "best_params.yaml"


def get_flair_optimized_pipeline_params_path(embedding_path: str, dataset_name: str) -> Path:
    return FLAIR_HPS_OUTPUT_PATH / embedding_path / dataset_name / "best_params.yaml"


OPTIMIZED_PIPELINES_PARAMS_PATH_FN = {
    "flair": get_flair_optimized_pipeline_params_path,
    "lightning": get_lightning_optimized_pipeline_params_path,
}


def get_optimized_pipeline_params_path(
    library: ModelExecutionLibrary, embedding_path: str, dataset_name: str
) -> Path:
    return OPTIMIZED_PIPELINES_PARAMS_PATH_FN[library](
        embedding_path=embedding_path, dataset_name=dataset_name
    )


def build_embedding_path_by_name(embedding_name: str) -> str:
    return embedding_name.replace("/", "__")


def get_embedding_name_by_path(embedding_path: str) -> str:
    return embedding_path.replace("__", "/")
