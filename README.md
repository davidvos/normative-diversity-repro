# Examining Sensitivity and Representativeness of Normative Diversity Metrics

This repository contains the supporting material for the paper 'Examining Sensitivity and Representativeness of Normative Diversity Metrics'. We reproduce the paper 'RADio* – An Introduction to Measuring Normative Diversity in News Recommendations'. The code in the 'dart/' folder comes from the original RADio implementation, [that can be found here](https://github.com/svrijenhoek/RADio). 

## Get Started

1. Download this repository
2. Open your Terminal in the directory of this repository
3. Run ```mkdir data```
4. [Download the required Data files here](https://www.dropbox.com/scl/fi/ywvdjb6g6fq9igdjz34cc/data.zip?rlkey=o1n90ipkkdrhslryjxf8xt401&st=a21hote7&dl=0)
5. Unzip the downloaded file, and move its contents to the ```data/``` folder.
6. [Download the required Results files here](https://www.dropbox.com/scl/fi/ue4fd24xzud9inuuicfvm/results.zip?rlkey=zgezncdv6wm4yfqvrictuz160&st=bitg4e4s&dl=0)
7. Unzip the downloaded file and move its contents to the ```results/``` folder.
8. Run ```mkdir data/MIND```
9. [Download the MIND dataset here](https://msnews.github.io/)
10. Unzip the downloaded file, and move its contents the the ```data/MIND/``` folder.

#### RADio on MIND

Run ```python compute_radio.py```

#### Analysis of Different User Samples

```Results.ipynb``` is a notebook containing code to process the results, do significance testing and visualize the metrics as the user sample size increases.

#### Simulation of Rankers

```Simulate.ipynb``` is a notebook containing the simulation of different rankers with a shared distribution as mentioned in 'Examining Sensitivity and Representativeness of Normative Diversity Metrics'. It also contains an analysis of the output of different news recommenders on the MIND dataset.

## Results

The Python scripts write their results to the ```results/``` folder. The Jupyter Notebook ```Results.ipynb``` contains the code necessary to generate the figures from the paper 'Examining Sensitivity and Representativeness of Normative Diversity Metrics'.

## Data Generation

By downloading the files in the 'Get Started' section, you have access to all data and recommendations necessary to run our experiments. If you want to generate the recommendation sets yourself, you can follow these instructions.

1. Run ```generate_baselines.py```
2. Use the [Recommenders Team](https://github.com/recommenders-team/recommenders) repository to train the NAML, NPA, NRMS and LSTUR models
3. Use the trained models to generate recommendations for the corresponding files in ```data/MIND/MINDLarge_dev```

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
