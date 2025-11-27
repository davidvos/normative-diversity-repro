# A Study of Normative Diversity Metrics in News Recommendations

This repository contains the supporting material for the paper 'A Study of Normative Diversity Metrics in News Recommendations'. We reproduce the paper 'RADio* – An Introduction to Measuring Normative Diversity in News Recommendations'. The code in the 'dart/' folder comes from the original RADio implementation, [that can be found here](https://github.com/svrijenhoek/RADio).

## 1. Environment Setup

1.  **Clone this repository**:
    ```bash
    git clone <repository_url>
    cd normative-diversity-repro
    ```

2.  **Install dependencies**:
    We recommend using `uv` for fast dependency management, but `pip` works as well.
    ```bash
    # Using pip
    pip install -e .

    # Using uv
    uv pip install -e .
    ```

3.  **Create data directories**:
    ```bash
    mkdir -p data results
    ```

## 2. Data Setup

You have two options: **Quick Start** (download pre-processed data) or **Manual Generation** (generate from scratch).

### Option A: Quick Start (Recommended)

1.  [Download the Data files here](https://drive.google.com/file/d/1bqKfXv2-nR7v1ve8Qqilkp-9-JRrym3T/view?usp=drive_link).
2.  Unzip the downloaded file.
3.  Move the contents to the `data/` folder. Your structure should look like:
    ```
    data/
    ├── mind/
    ├── ebnerd/
    └── adressa/
    ```

### Option B: Manual Generation

If you prefer to generate the data yourself:

1.  **Download Raw Data**:
    *   **MIND**: Download from [MIND News Dataset](https://msnews.github.io/).
    *   **EBNeRD**: Download from [EBNeRD RecSys](https://recsys.eb.dk/).
    *   **Adressa**: Access requires an agreement with the provider.

2.  **Generate Baselines**:
    Run the generation scripts to create baseline recommendations and files.
    ```bash
    python scripts/generate_baselines_mind.py
    python scripts/generate_baselines_ebnerd.py
    python scripts/generate_baselines_adressa.py
    ```

3.  **Train Models**:
    Use the [Recommenders Team](https://github.com/recommenders-team/recommenders) repository to train NAML, NPA, NRMS, and LSTUR models. Generate recommendations for the corresponding files in `data/mind/MINDlarge_dev`, `data/ebnerd/val`, and `data/adressa/val`.

4.  **Preprocess News Data**:
    Run the preprocessing scripts to enrich the news data (e.g., sentiment, complexity).
    ```bash
    python scripts/preprocess_mind_news.py
    python scripts/preprocess_ebnerd_news.py
    python scripts/preprocess_adressa_news.py
    ```

## 3. Running RADio (Basic Metrics)

To compute the standard RADio metrics (Calibration, Fragmentation, Activation, etc.) for a dataset:

```bash
# Run for MIND
python scripts/compute_radio.py --dataset mind --max-behaviors 1000

# Run for EBNeRD
python scripts/compute_radio.py --dataset ebnerd --max-behaviors 1000

# Run for Adressa
python scripts/compute_radio.py --dataset adressa --max-behaviors 1000
```

*   `--dataset`: `mind`, `ebnerd`, `adressa`, or `all`.
*   `--max-behaviors`: Limit the number of user behaviors to process (useful for testing). Default is `-1` (all).
*   `--metric`: Specific metric to compute (e.g., `activation`, `fragmentation`). Default is `all`.

## 4. Running with Re-rankers (Trade-offs)

To evaluate how re-ranking affects diversity metrics (trade-off curves), use the `--tradeoff-lambdas` flag. This will re-rank the top-k recommendations based on a linear combination of relevance and the diversity metric.

```bash
# Example: Evaluate trade-offs for all metrics with lambda values 0.0, 0.5, 1.0
python scripts/compute_radio.py --dataset mind --max-behaviors 1000 --tradeoff-lambdas "0.0,0.5,1.0"
```

*   `--tradeoff-lambdas`: Comma-separated list of lambda values (e.g., `"0.0,0.1,0.5,1.0"`).
*   `--rerank-all`: If set, re-ranks the entire candidate list instead of just the top-k.

### Production Reranker Simulation

To simulate a "production" reranker that uses predicted relevance scores (from the base recommender) instead of ground truth labels for the optimization, add the `--use-predicted-relevance` flag. This is useful for evaluating how the reranker would perform in a real-world setting where ground truth is unknown at inference time.

```bash
# Example: Production reranker simulation
python scripts/compute_radio.py --dataset mind --max-behaviors 1000 --tradeoff-lambdas "0.0,0.5" --use-predicted-relevance
```



## 5. Analysis

We provide notebooks to analyze and visualize the results:

*   `notebooks/MIND Results.ipynb`, `notebooks/EBNeRD Results.ipynb`, `notebooks/Adressa Results.ipynb`: Process results, perform significance testing, and visualize metric distributions.