# IVY League College Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Linear Regression model to predict the chance of admission to IVY league college

## See LIVE demo [here](https://ivy-league-collage-predictor.streamlit.app/)

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
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
|          
|__ EDA.ipynb          <- EDA notebook
|          
|__ src/train.py       <- training script
|
|__ streamlit_app.py   <- streamlit app
```

# Intro to the Dataset and the Aim
<img src="banner.jpg" alt="jamboree logo banner" style="width: 800px;"/>

An education institute has recently launched a dataset that contains the details of students who have applied for admission to IVY League College. The Jamboree team wants to know what factors are important for a student's success in getting into an IVY league college. They also want to see if we can make a predictive model to predict the chance of admission to IVY league college using the given features.

**Dataset**

This dataset contains the details of 500 students who have applied for admission to IVY League College along with their success rate.

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
1. To analyze what factors are important for a student's success in getting into an IVY league college.
2. To make a predictive interpretable model to predict the chance of admission (`chance_of_admit`) to IVY league college using the given features.

**Methods and Techniques used:** EDA, feature engineering, modeling using sklearn pipelines, hyperparameter tuning

**Measure of Performance and Minimum Threshold to reach the business objective**: MSE of 1% or less with max VIF less than 5

**Assumptions**
1. This fairly small dataset (500 entries) is representative of the real-world population.
2. The data is stable and does not change over time. Thus model is assumed to not decay. 

# Results
* The best model is Linear Regression with 
    * MSE: **`0.3421%`**
    * Accuracy(R^2) **`82.282%`**
    * VIF: **`4.26`**
* The following features with weights are selected by the model which signifies the importance of those features
Intercept: -0.25633387548676145

| Features            | Coefficients |
|---------------------|--------------|
| `cgpa`              | 0.599586     |
| `research`          | 0.205412     |
| `gre_score`         | 0.122329     |
| `toefl_score`       | 0.097854     |
| `university_rating` | 0.068220     |

*  CGPA has the most weight for predicting the chance of admit to IVY league college followed by research and GRE score


Check EDA under `/notebooks` for more details or see the Kaggle Notebook [here](https://www.kaggle.com/code/athuljyothis/ivy-league-collage-predictor-ml-model-82-acc)
--------

