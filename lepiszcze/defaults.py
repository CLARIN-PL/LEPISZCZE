from typing import Dict, Union


def get_model_checkpoint_kwargs() -> Dict[str, Union[str, None, bool]]:
    return {
        "filename": "last",
        "monitor": None,
        "save_last": False,
    }
