# Examining Sensitivity and Representativeness of Normative Diversity Metrics

This repository contains the supporting material for the paper 'Examining Sensitivity and Representativeness of Normative Diversity Metrics'. We reproduce the paper 'RADio* – An Introduction to Measuring Normative Diversity in News Recommendations'. The code in the 'dart/' folder comes from the original RADio implementation, [that can be found here](https://github.com/svrijenhoek/RADio). 

## Get Started

1. Clone this repository.
2. Open your Terminal in the directory of this repository.
3. Install dependencies: ```pip install -r requirements.txt```
4. Run ```mkdir data``` and ```mkdir results```.
5. [Download the Data & Results files here](https://www.dropbox.com/scl/fi/6uj6e3ji7kn8zkq4sr6s7/data_results.zip?rlkey=xrrrbgmy457bz45rcgu8434oc&st=0nt8y8ht&dl=0)
6. Unzip the downloaded file, and move its contents to the ```data/``` and ```results/``` folders.

#### RADio on MIND or EBNeRD

Run ```python compute_radio.py --dataset 'mind'``` or ```python compute_radio.py --dataset 'ebnerd'```

You can consider a maximum number of behaviors (e.g. 1000) by running ```python compute_radio.py --dataset 'mind' --max-behaviors 1000``` or ```python compute_radio.py --dataset 'ebnerd' --max-behaviors 1000```

#### Include greedy ranking

Run ```python compute_radio.py --dataset 'mind' --max-behaviors 1000 --greedy``` or ```python compute_radio.py --dataset 'ebnerd' --max-behaviors 1000 --greedy```

Be aware of the fact that greedy ranking is computationally expensive, and that running it on the full dataset is not feasible.

#### Analysis of Different User Samples

```MIND_Results.ipynb``` and ```EBNeRD_Results.ipynb``` are notebooks containing code to process the results, do significance testing and visualize distributions of the metrics.

```MIND_Analysis.ipynb``` and ```EBNeRD_Analysis.ipynb``` are notebooks containing the code to plot figures of the distribution of article attributes in recommendations.

## Data Generation

By downloading the files in the 'Get Started' section, you have access to all data and recommendations necessary to run our experiments. If you want to generate the recommendation sets yourself, you can follow these instructions.

1. Run ```generate_baselines_mind.py``` and ```generate_baselines_ebnerd.py```
2. Use the [Recommenders Team](https://github.com/recommenders-team/recommenders) repository to train the NAML, NPA, NRMS and LSTUR models
3. Use the trained models to generate recommendations for the corresponding files in ```data/mind/MINDlarge_dev``` and ```data/ebnerd/val```
4. Run ```python preprocess_ebnerd_news.py``` to preprocess the EBNeRD news data. 

# Additional Results

## Figures

### MIND Attribute Counts

<div>
    <img src="results/mind_categories.png">
    <img src="results/mind_subcategories.png">
    <img src="results/mind_activations.png">
</div>

### EBNeRD Attribute Counts

<div>
    <img src="results/ebnerd_categories.png">
    <img src="results/ebnerd_subcategories.png">
    <img src="results/ebnerd_activations.png">
</div>

### MIND Distributions

<div>
    <img src="results/mind_topic_calibrations_distributions.png">
    <img src="results/mind_subtopic_calibrations_distributions.png">
    <img src="results/mind_activations_distributions.png">
</div>

### EBNeRD Distributions

<div>
    <img src="results/ebnerd_topic_calibration_distributions.png">
    <img src="results/ebnerd_topic_subcalibration_distributions.png">
    <img src="results/ebnerd_activation_distributions.png">
</div>

### MIND Convergences

<div>
    <img src="results/mind_converging_topic_calibrations.png">
    <img src="results/mind_converging_subtopic_calibrations.png">
    <img src="results/mind_converging_complexity_calibrations.png">
    <img src="results/mind_converging_activations.png">
    <img src="results/mind_converging_tf_idf_ild_values.png">
    <img src="results/mind_converging_sentbert_ild_values.png">
    <img src="results/mind_converging_gini_values.png">
    <img src="results/mind_converging_ndcg_values.png">
</div>

### EBNeRD Convergences

<div>
    <img src="results/ebnerd_converging_topic_calibrations.png">
    <img src="results/ebnerd_converging_subtopic_calibrations.png">
    <img src="results/ebnerd_converging_complexity_calibrations.png">
    <img src="results/ebnerd_converging_activations.png">
    <img src="results/ebnerd_converging_tf_idf_ild_values.png">
    <img src="results/ebnerd_converging_sentbert_ild_values.png">
    <img src="results/ebnerd_converging_gini_values.png">
    <img src="results/ebnerd_converging_ndcg_values.png">
</div>

## Results on sample of MIND with available body texts

|                  |   topic_calibrations |   complexity_calibrations |   fragmentations |   activations |   representations |   ndcg_values |
|:-----------------|---------------------:|--------------------------:|-----------------:|--------------:|------------------:|--------------:|
| **Popular**              |             0.613066 |                  0.47361  |         0.591352 |      0.279849 |          0.232561 |      0.226078 |
| **Random**           |             0.612829 |                  0.473121 |         0.574202 |      0.280122 |          0.236508 |      0.230426 |
| **Original Random** |             0.629983 |                  0.499395 |         0.583904 |      0.319876 |          0.268683 |      0.228604 |
