# Normative Diversity in News Recommendations: A Reproducibility Study

This repository contains the supporting material for the paper 'Normative Diversity in News Recommendations: A Reproducibility Study'. We reproduce the paper 'RADio* – An Introduction to Measuring Normative Diversity in News Recommendations'. The code in the 'dart/' folder comes from the original RADio implementation, [that can be found here](https://github.com/svrijenhoek/RADio). 

## Get Started

1. Download this repository
1. Run ```mkdir data results```
2. [Download the required files here](https://www.dropbox.com/scl/fi/ywvdjb6g6fq9igdjz34cc/data.zip?rlkey=o1n90ipkkdrhslryjxf8xt401&st=8ipbz0lh&dl=0)
4. Unzip the downloaded file, and move its contents to the ```data/``` folder.
5. [Download the MIND dataset here](https://msnews.github.io/)
6. Unzip the downloaded file, and move its contents the the ```data/MIND/``` folder.

#### RADio on degraded MIND Small

Run ```python main_small.py```

#### Topic Calibration on MIND Large

Run ```python topic_calibration.py```

#### Topic Calibration in a Retrieval Setting

Run ```python topic_calibration_retrieval.py```

## Results

The three Python files ```main_small.py```, ```topic_calibration.py```, and ```topic_calibration_retrieval.py``` write their results to the ```results/``` folder. The Jupyter Notebook ```Results.ipynb``` contains the code necessary to generate the figures from the paper 'Normative Diversity in News Recommendations: A Reproducibility Study'.

## Data Generation

By downloading the files in the 'Get Started' section, you have access to all data and recommendations necessary to run our experiments. If you want to generate the recommendation sets yourself, you can follow these instructions.

1. Run ```extend_candidates.py```
2. Run ```generate_baselines.py```
3. Use the [Recommenders Team](https://github.com/recommenders-team/recommenders) repository to train the NAML, NPA, NRMS and LSTUR models
4. Use the trained models to generate recommendations for the corresponding files in ```data/candidates```