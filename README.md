# LEPISZCZE
This is the official code implementation for the LEPISZCZE benchmark experiments  
**"This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish"**
(Łukasz Augustyniak, Kamil Tagowski, Albert Sawczyn, 
Denis Janiak, Roman Bartusiak, Adrian Szymczak, 
Marcin Wątroba, Arkadiusz Janz, Piotr Szymański, 
Mikołaj Morzy, Tomasz Kajdanowicz, Maciej Piasecki).

## Resources

LEPISZCZE benchmark resources

| Name                  | Description                                                                                                                                      | URL                                              |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| Libary                | **clarin-pl/embeddings** Our library with pre-defined NLP pipelines for text classification, pair text classification and sequence labeling taks | [GitHub](https://github.com/cLARIN-PL/embeddings/) |
| Experiments dashboard | Weight&Biases dashboard with our experiments                                                                                                     | [W&B](https://wandb.ai/embeddings/LEPISZCZE)     |
| Leaderboard           | LEPISZCZE Leaderboard                                                                                                                            | TBA                                              |
 | Datasets              | LEPISZCZE Datasets are accessible through our HuggingFace Hub organization page.                                                                 | [HuggingFace](https://huggingface.co/clarin-pl) | 
 | KLEJ-Datasets         | Datasets for KLEJ benchmark are accessible through Allegro HuggingFace organization page.                                                        | [HuggingFace](https://huggingface.co/allegro) ||               |                                                                                                                                                  |                                                    |


## Citation

TBA

## Installation

Repository can be setup via [poetry](https://python-poetry.org) or via [docker](https://www.docker.com). 

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

Our experiments can be easily reproducible with [DVC](https://dvc.org) repro & [W&B](https://wandb.ai) logging. Using `dvc repro` command and with W&B token setup. 

**DISCLAIMER** Reproduction of full pipeline could take above 2000 hours to compelete on a single GPU device. We advise to execute stages in parallel on mutiple GPU  computing devices.

### Access to DVC remote repository

Due to the size of pipeline outputs data, we do not provide public access to our DVC Remote Repository. However, if you are interested in any kinds of data artifacts, don't hesitate to get in touch with us:
- Łukasz <lukasz.augustyniak@pwr.edu.pl>
- Kamil <kamil.tagowski@pwr.edu.pl>

## Experiments

Experiments configs can be found under [configs](experiments/configs)

**DISCLAIMER** For some of the dataset we had to limit manually maximum sequence length to `512` for Hyper Parameter Search.

Models hyperparameters configuations can be accessed via W&B dashboard. Example: [[LINK]](https://wandb.ai/embeddings/LEPISZCZE/runs/2v6gyxjy/overview?workspace=user-)

### Datasets configurations


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
| [allegro/klej-cdsc-e](https://huggingface.co/datasets/allegro/klej-cdsc-e)                    | pair classification (textual entailment)*        | (sentence_A, sentence_B) | entailment_judgment | The polish sentence pairs which are human-annotated for textualentailment.                                                                                                                |
