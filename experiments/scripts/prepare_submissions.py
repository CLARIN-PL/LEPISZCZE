import typer
from embeddings.evaluator.submission import AveragedSubmission
from tqdm.auto import tqdm

from lepiszcze import DATASET_TASK_MAPPING, ModelExecutionLibrary
from lepiszcze.paths import (
    PIPELINES_OUTPUT_PATHS_MAPPING,
    SUBMISSIONS_PATH,
    get_optimized_pipeline_params_path,
)

app = typer.Typer()


def run(
    model: str = typer.Option(..., help="Model"),
    library: ModelExecutionLibrary = typer.Option(..., help="Library"),
) -> None:
    model_output_path = PIPELINES_OUTPUT_PATHS_MAPPING[library] / model

    dataset_paths = [it for it in model_output_path.iterdir() if it.is_dir()]
    pbar = tqdm(dataset_paths, desc="Dataset")
    for dataset_path in pbar:
        dataset = dataset_path.name
        pbar.set_description(f"Dataset {dataset}")

        model_ds_output_path = model_output_path / dataset
        best_params_path = get_optimized_pipeline_params_path(
            library=library, embedding_path=model, dataset_name=dataset
        )
        run_dirs = [
            it for it in model_ds_output_path.iterdir() if ("run" in str(it)) and it.is_dir()
        ]
        if len(run_dirs) > 0:
            wandb_configs_path = []
            evaluation_file_paths = []
            packages_file_paths = []

            for run_dir in run_dirs:
                evaluation_file_path = run_dir / "evaluation.json"
                packages_file_path = run_dir / "packages.json"
                wandb_config_path = run_dir / "config.yaml"

                assert evaluation_file_path.exists()
                assert packages_file_path.exists()
                assert wandb_config_path.exists()

                evaluation_file_paths.append(evaluation_file_path)
                packages_file_paths.append(packages_file_path)
                wandb_configs_path.append(wandb_config_path)

            submission_name = f"{dataset}_{model}"
            submission = AveragedSubmission.from_local_disk(
                submission_name=submission_name,
                task=DATASET_TASK_MAPPING[dataset],
                evaluation_file_paths=evaluation_file_paths,
                packages_file_paths=packages_file_paths,
                wandb_config_paths=wandb_configs_path,
                best_params_path=best_params_path,
            )
            submission.save_json(SUBMISSIONS_PATH / model)


typer.run(run)
