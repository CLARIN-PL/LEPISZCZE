from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def convert_float_to_str(x: float) -> str:
    return f"{x:.2f}"


def percentage_style(x: Union[float, Tuple[float, float]]) -> Union[float, Tuple[float, float]]:
    """Percantage style fn."""
    if isinstance(x, float):
        return round(x * 100, 2)
    elif isinstance(x, tuple):
        return round(x[0] * 100, 2), round(x[1] * 100, 2)
    raise ValueError("X parsing error")


def get_top_score_bold(x: pd.Series) -> List[str]:
    max_id = np.argmax(x.values)

    output = []
    for i in range(len(x)):
        if i == max_id:
            out_str = f'$\\mathbf{{{convert_float_to_str(x[i][0]) + " ± " + convert_float_to_str(x[i][1])}}}'
            out_str += "$"
            output.append(out_str)

        elif -1_000_000 < x[i][0] < 1_000_000:
            out_str = f'${convert_float_to_str(x[i][0]) + " ± " + convert_float_to_str(x[i][1])}'
            out_str += "$"
            output.append(out_str)

        else:
            output.append("$\\times$")
    return output
