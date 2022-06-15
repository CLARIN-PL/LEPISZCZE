def get_task_run_name(embedding_path: str, dataset: str, run_id: int) -> str:
    return f"{dataset}_{embedding_path}_run_{run_id}"


def get_hps_run_name(embedding_path: str, dataset: str) -> str:
    return f"hps_{dataset}_{embedding_path}"
