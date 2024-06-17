# IVY League Collage Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Linear Regresssion model to predict the chance of admission to IVY league college

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for jamboree_admission_linear_regression
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── jamboree_admission_linear_regression                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes jamboree_admission_linear_regression a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

## Intro to the Dataset and the Aim
\<img src="/jamboree_logo.png" alt="jamboree logo banner" style="width: 800px;"/>

Jamboree has helped thousands of students like you make it to top colleges abroad. Be it GMAT, GRE or SAT, their unique problem-solving methods ensure maximum scores with minimum effort.

Jamboree team wants to know what factors are important for a students success in getting into an IVY league college. They also want to see if we can make a predictive model to predict the chance of admission to IVY league college using the given features.

**Dataset**

This dataset contains the details of 500 students who have applied for admission to IVY league college along with their success rate.

Summary of sanitized data:

| Column              | Description         | 
|---------------------|---------------------|
| `serial_no`         | Unique row ID       |
| `gre_score`         | Out of 340          |
| `toefl_score`       | Out of 120          |
| `university_rating` | Out of 5            | 
| `sop`               | Out of 5            | 
| `lor`               | Out of 5            | 
| `cgpa`              | Out of 10           | 
| `research`          | Either 0 or 1       |
| `chance_of_admit`   | Ranging from 0 to 1 |

Additional feature engineered columns:

| Column                | Description                                    |
|-----------------------|------------------------------------------------|
| `gre_sqr`             | Square of `gre_score`                          |
| `cgpa_sqr`            | Square of `cgpa`                               |
| `uni_rating_sqr`      | Square of `university_rating`                  |
| `gre_uni_ratio`       | `gre_score`/`university_rating`                |
| `cgpa_uni_ratio`      | `cgpa`/`university_rating`                     |
| `gre_cgpa_prod`       | `gre_score`*`cgpa`                             |
| `gre_avg_uni_rating`  | Avg `gre_score` grouped by `university_rating` |
| `cgpa_avg_uni_rating` | Avg `cgpa` grouped by `university_rating`      |



**Aim:** 
1. To anlyze what factors are important for a students success in getting into an IVY league college.
2. To make a predictive model to predict the chance of admission (`chance_of_admit`) to IVY league college using the given features.

**Methods and Techniques used:** EDA, feature engineering, modeling using sklearn pipelines, hyperparameter tuning

**Measure of Performance and Minimum Threshold to reach the business objective** : MSE of 5% or less

**Assumptions**
1. This fairly small dataset (500 entries) is representative of the real world population.
2. The data is stable and does not change over time. Thus model assumed to not decay. 


# Results
* The best model is Linear Regression with **MSE of 0.330%** and **R2 of 0.828**
* The following features with weights are selected by the model which signifies the importance of those features
Intercept: -0.39867413457616596

| Features          | Coefficients |
|-------------------|--------------|
| cgpa              | 0.505760     |
| gre_score         | 0.192355     |
| research          | 0.160473     |
| toefl_score       | 0.121719     |
| lor               | 0.061737     |
| university_rating | 0.040744     |
--------

