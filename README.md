# LEPISZCZE
This is the official code implementation for the LEPISZCZE benchmark experiments.
**"This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish"** (NeurIPS 2022)
(Łukasz Augustyniak, Kamil Tagowski, Albert Sawczyn, 
Denis Janiak, Roman Bartusiak, Adrian Szymczak, 
Marcin Wątroba, Arkadiusz Janz, Piotr Szymański, 
Mikołaj Morzy, Tomasz Kajdanowicz, Maciej Piasecki).

## Resources

LEPISZCZE benchmark resources

| Name                  | Description                                                                                                                                      | URL                                                |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| **Leaderboard**           | **Interactive results across all LEPISZCZE tasks, including extractive QA**                                                                           | [**LEPISZCZE**](https://clarin-pl.github.io/LEPISZCZE/leaderboard/) |
| Library               | **clarin-pl/embeddings** Our library with predefined NLP pipelines for text classification, pair classification and sequence labeling tasks       | [GitHub](https://github.com/CLARIN-PL/embeddings/) |
| Experiments dashboard | Weights & Biases dashboard with our experiments                                                                                                  | [W&B](https://wandb.ai/embeddings/LEPISZCZE)       |
 | Datasets              | LEPISZCZE Datasets are accessible through our HuggingFace Hub organization page.                                                                 | [HuggingFace](https://huggingface.co/clarin-pl)    | 
 | KLEJ-Datasets         | Datasets for the KLEJ benchmark are accessible through the Allegro Hugging Face organization page.                                               | [HuggingFace](https://huggingface.co/allegro)      |

### Updating the leaderboard

The GitHub Pages leaderboard uses a minimal, checked-in snapshot rather than loading mutable
cross-repository data at runtime. To rebuild it from the historical results:

```bash
git clone https://github.com/CLARIN-PL/embeddings.git
cd LEPISZCZE
python3 scripts/build_leaderboard.py \
  --source-dir ../embeddings/webpage/data/results \
  --source-commit "$(git -C ../embeddings rev-parse HEAD)"
python3 scripts/build_leaderboard.py --check docs/data/leaderboard.json
```

Commit the generated `docs/data/leaderboard.json`. GitHub Actions validates its schema before
deploying the static site.

### Legal AI dataset pipeline

LEPISZCZE maintains a public registry of datasets developed through two active Legal AI projects:

- [JuDDGES](https://github.com/pwr-ai/juddges-app) — judicial-decision gathering, encoding,
  human-in-the-loop annotation, structured information extraction and evaluation for Polish and
  England & Wales case law.
- [AITAX](https://ai.pwr.edu.pl/projects/aitax) — Polish tax-document retrieval, authority-position
  summarization and grounded drafting of tax-interpretation requests.

The registry distinguishes source corpora, enriched corpora, instruction datasets and benchmark
datasets. It also separates public datasets, public previews and benchmark candidates from official
leaderboard components. A dataset is promoted only after its version and test split are frozen,
licensing and leakage review are complete, metrics and baselines are published, and evaluation is
reproducible from a versioned protocol.

Registry data lives in `docs/data/datasets.json` and is validated during every Pages deployment:

```bash
python3 scripts/check_datasets.py docs/data/datasets.json
```


## Citation

```bibtex
@inproceedings{augustyniak2022lepiszcze,
 author = {Augustyniak, Lukasz and Tagowski, Kamil and Sawczyn, Albert and Janiak, Denis and Bartusiak, Roman and Szymczak, Adrian and Janz, Arkadiusz and Szyma\'{n}ski, Piotr and W\k{a}troba, Marcin and Morzy, Miko\l aj and Kajdanowicz, Tomasz and Piasecki, Maciej},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {21805--21818},
 publisher = {Curran Associates, Inc.},
 title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/890b206ebb79e550f3988cb8db936f42-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}

```

## Contact

In case of any question or concerns about LEPISZCZE benchmark feel free to contact us:
- Łukasz <lukasz.augustyniak@pwr.edu.pl>
- Kamil <kamil.tagowski@pwr.edu.pl>
- Albert <albert.sawczyn@pwr.edu.pl>
- Denis <denis.janiak@pwr.edu.pl>

**DVC Repository Access** Due to the size of pipeline outputs data, we do not provide public access to our DVC Remote Repository. However, if you are interested in any kinds of data artifacts, don't hesitate to get in touch with us.
## Installation

The repository can be set up via [Poetry](https://python-poetry.org) or [Docker](https://www.docker.com).

### Requirements installation and environment setup via poetry 

Prerequisites: 
- **Python: 3.9+**
- **Poetry**  [[LINK]](https://python-poetry.org).
- **CUDA 11.3+ for GPU support (Recommended)**

Installation
```bash
poetry install
```

For GPU support 
```bash
poetry run poe force-torch-cuda
```

### Using docker image

Building image
```bash
docker build . -f docker/Dockerfile -t LEPISZCZE
```

After the container setup use [conda](http://conda.io) env `LEPISZCZE`
```bash
conda activate LEPISZCZE
```

## Reproducibility

Our experiments can be easily reproduced with [DVC](https://dvc.org) repro & [W&B](https://wandb.ai) logging. Using `dvc repro` command and with W&B token setup. 

**DISCLAIMER** Reproducing the full pipeline can take more than 2,000 hours on a single GPU. We recommend executing stages in parallel across multiple GPUs.


## Experiments

Experiments configs can be found under [configs](experiments/configs)

**DISCLAIMER** For some of the dataset we had to limit manually maximum sequence length to `512` for Hyper Parameter Search.

Model hyperparameter configurations can be accessed via the W&B dashboard. Example: [[LINK]](https://wandb.ai/embeddings/LEPISZCZE/runs/2v6gyxjy/overview?workspace=user-)

### Datasets configurations

The extractive QA datasets hosted under the `expansio` Hugging Face namespace are maintained as
part of LEPISZCZE. They extend the benchmark beyond the dataset set described in the original
NeurIPS 2022 paper.


| dataset name                                                                                  | task type                                          | input_column_name(s) | target_column_name | description                                                                                                                                                                               |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------|----------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [clarin-pl/kpwr-ner](https://huggingface.co/datasets/clarin-pl/kpwr-ner)                      | sequence labeling (named entity recognition)       | tokens               | ner       | KPWR-NER is a part of the  Polish Corpus of Wrocław University of Technology (KPWr). Its objective is recognition of named entities, e.g., people, institutions etc.                      |
| [clarin-pl/polemo2-official](https://huggingface.co/datasets/clarin-pl/polemo2-official )       | classification  (sentiment analysis)             | text                 | target    | A corpus of consumer reviews from 4 domains: medicine, hotels, products and school.                                                                                                       |
| [clarin-pl/2021-punctuation-restoration](https://huggingface.co/datasets/clarin-pl/2021-punctuation-restoration)                      | punctuation restoration                           | text_in              | text_out  | Dataset contains original texts and ASR output. It is a part of PolEval 2021 Competition.                                                                                                 |
| [clarin-pl/nkjp-pos](https://huggingface.co/datasets/clarin-pl/nkjp-pos)                      | sequence labeling (part-of-speech tagging)       | tokens               | pos_tags  | NKJP-POS is a part of the National Corpus of Polish. Its objective is part-of-speech tagging, e.g., nouns, verbs, adjectives, adverbs, etc.                                               |
| [clarin-pl/aspectemo](https://huggingface.co/datasets/clarin-pl/aspectemo)                      | sequence labeling (sentiment classification)     | tokens               | labels    | AspectEmo Corpus is an extended version of a publicly available PolEmo 2.0 corpus of Polish customer reviews used in many projects on the use of different methods in sentiment analysis. |
| [laugustyniak/political-advertising-pl](https://huggingface.co/datasets/laugustyniak/political-advertising-pl)                      | sequence labeling (political advertising )                         | tokens               | tags      | First publicly open dataset for detecting specific text chunks and categories of political advertising in the Polish language.                                                            |
| [laugustyniak/abusive-clauses-pl](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl)                      | classification  (abusive-clauses)                           | text                  | class     | Dataset with Polish abusive clauses examples.                                                                                                                                             |
| [allegro/klej-dyk](https://huggingface.co/datasets/allegro/klej-dyk)                             | pair classification (question answering)*        | (question, answer)   | target    | The Did You Know (pol. Czy wiesz?) dataset consists of human-annotated question-answer pairs.                                                                                             |
| [allegro/klej-psc](https://huggingface.co/datasets/allegro/klej-psc)                             | pair classification (text summarization)*        | (extract_text, summary_text) | label     | The Polish Summaries Corpus contains news articles and their summaries.                                                                                                                   |
| [allegro/klej-cdsc-e](https://huggingface.co/datasets/allegro/klej-cdsc-e)                    | pair classification (textual entailment)*        | (sentence_A, sentence_B) | entailment_judgment | The polish sentence pairs which are human-annotated for textual entailment.                                                                                                                |
| [expansio/qa-wikipedia](https://huggingface.co/datasets/expansio/qa-wikipedia) | extractive question answering (SQuAD 2.0 style) | (question, context)  | answers   | Polish extractive QA dataset built on Polish Wikipedia passages. Includes paraphrased and unanswerable questions. ([DOI: 10.57967/hf/8855](https://doi.org/10.57967/hf/8855))                                            |
| [expansio/qa-nkjp](https://huggingface.co/datasets/expansio/qa-nkjp) | extractive question answering (SQuAD 2.0 style) | (question, context)  | answers   | Polish extractive QA dataset built on passages from the National Corpus of Polish (NKJP). Includes paraphrased and unanswerable questions. ([DOI: 10.57967/hf/8856](https://doi.org/10.57967/hf/8856))                  |
| [expansio/qa-kpwr](https://huggingface.co/datasets/expansio/qa-kpwr) | extractive question answering (SQuAD 2.0 style) | (question, context)  | answers   | Polish extractive QA dataset built on passages from the KPWr corpus (Polish Corpus of Wrocław University of Technology). Includes paraphrased and unanswerable questions. ([DOI: 10.57967/hf/8854](https://doi.org/10.57967/hf/8854)) |
