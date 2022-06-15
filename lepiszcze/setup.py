from datasets import disable_caching
from embeddings.utils.loggers import LightningLoggingConfig


def disable_hf_datasets_caching() -> None:
    # Disable generation of datasets cache files
    disable_caching()  # type: ignore


def get_lightning_logging_config(dataset_name: str) -> LightningLoggingConfig:
    return LightningLoggingConfig(
        loggers_names=["wandb"], tracking_project_name=dataset_name, wandb_entity="embeddings"
    )
