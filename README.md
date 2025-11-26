# A Study of Normative Diversity Metrics in News Recommendations

This repository contains the supporting material for the paper 'A Study of Normative Diversity Metrics in News Recommendations'. We reproduce the paper 'RADio* – An Introduction to Measuring Normative Diversity in News Recommendations'. The code in the 'dart/' folder comes from the original RADio implementation, [that can be found here](https://github.com/svrijenhoek/RADio). 

## Get Started

1. Clone this repository.
2. Open your Terminal in the directory of this repository.
3. Install the package in editable mode: ```pip install -e .```
4. Run ```mkdir data``` and ```mkdir results```.
5. [Download the Data files here](https://drive.google.com/file/d/1bqKfXv2-nR7v1ve8Qqilkp-9-JRrym3T/view?usp=drive_link)
6. Unzip the downloaded file, and move its contents to the ```data/``` folder.

#### RADio on MIND or EBNeRD

Run ```python scripts/compute_radio.py --dataset 'mind'``` or ```python scripts/compute_radio.py --dataset 'ebnerd'``` (use `'adressa'` for Adressa, or `'all'` to process every dataset sequentially). You can limit the workload with ```--max-behaviors```, e.g. ```python scripts/compute_radio.py --dataset 'mind' --max-behaviors 1000```.

#### Include greedy ranking

Run ```python scripts/compute_radio.py --dataset 'mind' --max-behaviors 1000 --greedy``` or ```python scripts/compute_radio.py --dataset 'ebnerd' --max-behaviors 1000 --greedy```

Be aware of the fact that greedy ranking is computationally expensive, and that running it on the full dataset is not feasible.

#### Complexity Calibration Variants

Each dataset now stores two complexity signals per article: a readability score (Flesch–Kincaid for MIND, LIX for EBNeRD/Adressa) and a lexical richness score (MTLD). `scripts/compute_radio.py` defaults to the readability signal, but you can switch to lexical richness via `--complexity-source mtld`.

To compute both complexity variants in a single pass, set `--complexity-source both`. The script will run twice per dataset (readability + MTLD) and automatically append `_readability`/`_mtld` to the output filenames (after any custom suffix you provide). You can also supply `--output-suffix _experiment` to tag all generated files when running individual configurations.

#### Metric Sweeps

When you only care about a single metric and want to evaluate every design choice for it, use `--metric {complexity,activation,fragmentation}`. The script automatically sweeps the relevant options (complexity signals, sentiment sources, or story thresholds), skips computing the other metrics (aside from NDCG), and writes a single aggregated results file per dataset/metric (e.g. `results/mind_activation_sweep_k@10.pkl`). Example: `python scripts/compute_radio.py --dataset mind --metric activation` evaluates lexicon, transformer, and dataset sentiments in separate runs and saves the joined results plus combined trade-off curves. Combine this with `--dataset all --metric activation` to sweep every dataset. For these sweeps the corresponding `--complexity-source`, `--sentiment-source`, or `--story-threshold` flags are ignored, and the sweep outputs omit the `ndcg_values` column to keep files focused on the requested metric.

#### Story Clustering Thresholds

Fragmentation relies on TF-IDF story clustering with cosine similarity thresholds of 0.1, 0.2, 0.3, 0.4, or 0.5 (and a 3-day time window). Choose the desired granularity via `--story-threshold` when running `scripts/compute_radio.py`. EBNeRD additionally keeps the dataset-provided `topics` label in `story_topics` (and `story_dataset`), and Fragmentation sweeps now include a `story_threshold=dataset` variant that evaluates those original groupings alongside the TF-IDF thresholds.

#### Sentiment Estimation Variants

Activation can now draw from either the dataset-specific lexicon sentiment (`sentiment_lexicon`) or a contextual multilingual transformer score (`cardiffnlp/twitter-xlm-roberta-base-sentiment`). Both columns are generated during preprocessing, so you can switch at runtime with `--sentiment-source {lexicon,transformer}` on `scripts/compute_radio.py`. EBNeRD additionally ships a `sentiment_score` field, exposed as `sentiment_dataset`, so activation sweeps include a `dataset` option only for EBNeRD.

#### Analysis of Different User Samples

```notebooks/MIND Results.ipynb```, ```notebooks/EBNeRD Results.ipynb``` and ```notebooks/Adressa Results.ipynb``` are notebooks containing code to process the results, do significance testing and visualize distributions of the metrics.

```notebooks/MIND Analysis.ipynb``` and ```notebooks/EBNeRD Analysis.ipynb``` are notebooks containing the code to plot figures of the distribution of article attributes in recommendations.

## Data Generation

By downloading the files in the 'Get Started' section, you have access to all data and recommendations necessary to run our experiments. If you want to generate the recommendation sets yourself, you can follow these instructions.

1. Download the MIND and EBNeRD datasets from the [MIND](https://msnews.github.io/) and [EBNeRD](https://recsys.eb.dk/) repositories. Access to Adressa dataset requires an agreement with the provider.
2. Run ```python scripts/generate_baselines_mind.py```, ```python scripts/generate_baselines_ebnerd.py``` and ```python scripts/generate_baselines_adressa.py```
3. Use the [Recommenders Team](https://github.com/recommenders-team/recommenders) repository to train the NAML, NPA, NRMS and LSTUR models
4. Use the trained models to generate recommendations for the corresponding files in ```data/mind/MINDlarge_dev```, ```data/ebnerd/val``` and ```data/adressa/val```
5. Run ```python scripts/preprocess_mind_news.py```, ```python scripts/preprocess_ebnerd_news.py``` and ```python scripts/preprocess_adressa_news.py``` to preprocess the news data. 