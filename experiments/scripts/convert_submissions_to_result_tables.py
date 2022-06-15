import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Set

import pandas as pd
import typer

from lepiszcze.df_utils import get_top_score_bold, percentage_style
from lepiszcze.paths import SUBMISSIONS_PATH, TABLES_PATH
from lepiszcze.submission_postprocessing import ValueExtractor, parse_json_file

app = typer.Typer()
KLEJ_DATASETS: Set[str] = {"cdsc_e", "dyk", "polemo2_in", "polemo2_out", "psc"}
MODEL_NAME_MAPPING: Dict[str, str] = {
    "allegro/herbert-base-cased": "HerBERT (base, cased)",
    "allegro/herbert-large-cased": "HerBERT (large, cased)",
    "dkleczek/bert-base-polish-cased-v1": "BERT (base, cased)",
    "dkleczek/bert-base-polish-uncased-v1": "BERT (base, uncased)",
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1": "XLM-RoBERTa (paraphrase)",
}
DATASET_NAME_MAPPING: Dict[str, str] = {
    "abusive_clauses": "Abusive Clauses",
    "aspectemo": "AspectEmo",
    "cdsc_e": "CDSC-E",
    "dyk": "DYK",
    "kpwr_ner": "KPWr NER",
    "nkjp_pos": "NKJP POS",
    "polemo2": "PolEmo 2.0",
    "polemo2_in": "PolEmo 2.0 (In-Domain)",
    "polemo2_out": "PolEmo 2.0 (Out-Domain)",
    "political_advertising": "Political Advertising",
    "punctuation_restoration": "Punctuation Restoration",
    "psc": "PSC",
}


def get_all_submissions_zip_list() -> List[Path]:
    submission_dirs = list(SUBMISSIONS_PATH.iterdir())

    zip_files = []
    for path in submission_dirs:
        path_zip_files = list(path.rglob("*.zip"))
        zip_files += [it for it in path_zip_files if "predictions" not in str(it)]
    return zip_files


def postprocess_dataframe(df: pd.DataFrame, mark_klej_datasets: bool = True) -> pd.DataFrame:
    df = df.pivot(index="model_name", columns="dataset_name", values="metrics_tuple")
    df = df.applymap(percentage_style)
    df = df.apply(get_top_score_bold)
    df = df.T
    df.index.name = ""
    df.columns.name = ""
    df.rename(columns=MODEL_NAME_MAPPING, inplace=True)

    dataset_name_mapping = DATASET_NAME_MAPPING
    if mark_klej_datasets:
        dataset_name_mapping = {
            k: f"{v}<KLEJ>" if k in KLEJ_DATASETS else v for k, v in DATASET_NAME_MAPPING.items()
        }
    df.rename(index=dataset_name_mapping, inplace=True)
    return df


def main(
    ignore_klej_datasets: bool = typer.Option(
        False, help="Whether to ignore datasets from KLEJ benchmark"
    )
) -> None:
    temp_dir = TemporaryDirectory()

    for path in get_all_submissions_zip_list():
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(temp_dir.name)

    details_list = [
        parse_json_file(path, ValueExtractor(metric_name="f1_macro"))
        for path in Path(temp_dir.name).rglob("*.json")
    ]
    df = pd.DataFrame(details_list)
    if ignore_klej_datasets:
        df = df[~df.dataset_name.isin(KLEJ_DATASETS)]

    df = postprocess_dataframe(df)

    TABLES_PATH.mkdir(exist_ok=True, parents=True)
    df.to_pickle(TABLES_PATH / "results.pkl")
    latex_table = (
        df.style.to_latex()
        .replace("NaN", "---")
        .replace("±", r"\pm")
        .replace(r"\$", "$")
        .replace("textbackslash ", "")
        .replace("\{", "{")
        .replace(r"\}", "}")
        .replace("_", "\_")
        .replace("<KLEJ>", "\\textcolor{red}{*}")
    )
    with open(TABLES_PATH / "results.tex", "w+") as f:
        f.write(latex_table)

    temp_dir.cleanup()


typer.run(main)
