# Production Reranker Usage Guide

## Overview

The production reranker (`rerank_production_topk`) balances relevance and diversity **without requiring ground truth labels**. However, you can still evaluate its performance using ground truth labels for visualization and analysis.

## Key Distinction

### During Optimization
- **Production reranker**: Uses **predicted relevance** (inferred from baseline rankings) to optimize the tradeoff
- **Research reranker** (`rerank_tradeoff_topk`): Uses **ground truth relevance** to optimize the tradeoff

### During Evaluation (both)
- Both can be evaluated using **ground truth NDCG** and diversity metrics

## Running the Evaluation

```bash
# Quick test run (10 behaviors, 3 lambda values)
uv run python evaluate_production_reranker.py --max-behaviors 10 --lambdas "0.0,0.5,1.0"

# Full evaluation with defaults
uv run python evaluate_production_reranker.py

# Custom configuration
uv run python evaluate_production_reranker.py \
  --dataset mind \
  --recommender nrms \
  --max-behaviors 100 \
  --lambdas "0.0,0.2,0.4,0.5,0.6,0.8,1.0"
```

This generates CSV files for visualization in notebooks:
- `results/mind_topic_production_tradeoff_k@10.csv`
- `results/mind_subtopic_production_tradeoff_k@10.csv`
- `results/mind_activation_production_tradeoff_k@10.csv`
- `results/mind_complexity_production_tradeoff_k@10.csv`

## Visualizing Results

You can use the same plotting code as `MIND Results.ipynb` but change the filename:

```python
# In your notebook
import pandas as pd
import matplotlib.pyplot as plt

# Load production reranker results instead of research reranker
tradeoff_df = pd.read_csv('results/mind_topic_production_tradeoff_k@10.csv')

# Same plotting code as before
fig, ax = plt.subplots(figsize=(10, 10))
for recommender, group in tradeoff_df.groupby('recommender'):
    ax.plot(group['divergence'], group['ndcg'], marker='o', label=recommender)
    
ax.set_xlabel('Topic Calibration')
ax.set_ylabel('NDCG@10')
ax.legend()
plt.show()
```

## Comparison: Production vs Research Rerankers

| Aspect | Production Reranker | Research Reranker |
|--------|---------------------|-------------------|
| **Optimizes using** | Predicted relevance (from rankings) | Ground truth relevance |
| **Use case** | Live systems, deployment | Offline evaluation, research |
| **Evaluation** | Can use ground truth NDCG | Uses ground truth NDCG |
| **Returns** | Single reranked list per lambda | Tradeoff curve dict |
| **Computational cost** | O(k²n) per lambda | O(k²n) per lambda |

## What This Means

**Yes, you can visualize the trade-off between ground truth relevance and diversity** even though the production reranker optimizes based on predicted relevance!

The workflow:
1. Production reranker uses predicted scores to decide what to rerank
2. We evaluate the reranked results using ground truth labels
3. This shows how well the predicted-optimization translates to actual performance

This is actually more realistic than the research version, since in production you'd never have ground truth labels available when making recommendations.

