from enum import Enum

from embeddings.metric.sequence_labeling import EvaluationMode, TaggingScheme


class SequenceLabelingEvaluationMode(str, Enum):
    conll = "conll"
    strict = "strict"
    unit = "unit"


SEQUENCE_LABELING_EVALUATION_MODES_ARGS = {
    "conll": {
        "evaluation_mode": EvaluationMode.CONLL,
        "tagging_scheme": None,
    },
    "strict": {
        "evaluation_mode": EvaluationMode.STRICT,
        "tagging_scheme": TaggingScheme.IOB2,
    },
    "unit": {
        "evaluation_mode": EvaluationMode.UNIT,
        "tagging_scheme": None,
    },
}
