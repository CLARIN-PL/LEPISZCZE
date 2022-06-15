__version__ = "0.1.0"

from enum import Enum
from typing import Dict


class ModelExecutionLibrary(str, Enum):
    flair = "flair"
    lightning = "lightning"
    sklearn = "sklearn"


DATASET_TASK_MAPPING: Dict[str, str] = {
    "abusive_clauses": "text_classification",
    "aspectemo": "sequence_labeling",
    "cdsc_e": "text_classification",
    "dyk": "text_classification",
    "kpwr_ner": "sequence_labeling",
    "polemo2": "text_classification",
    "polemo2_in": "text_classification",
    "polemo2_out": "text_classification",
    "political_advertising": "sequence_labeling",
    "psc": "text_classification",
    "punctuation_restoration": "sequence_labeling",
    "nkjp_pos": "sequence_labeling",
}
