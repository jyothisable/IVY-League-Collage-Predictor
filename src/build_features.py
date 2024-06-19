import pandas as pd
import numpy as np

def make_col_names(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function to rename columns after make_pipeline is applied
    '''
    return df.rename(lambda col: col.split('__')[-1],axis='columns')

def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    # df['gre_sqr'] = df['gre_score']**2
    # df['cgpa_sqr'] = df['cgpa']**2
    # df['uni_rating_sqr'] = df['university_rating']**2
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # df['gre_uni_ratio'] = df['gre_score']/df['university_rating']
    # df['cgpa_uni_ratio'] = df['cgpa']/df['university_rating']
    return df

def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    # df['gre_cgpa_prod'] = df['gre_score']*df['cgpa']
    # df['gre_cgpa_toefl_prod'] = df['gre_score']*df['toefl_score']*df['cgpa']
    return df

def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    # df['gre_avg_uni_rating'] = df['gre_score'].groupby(df['university_rating']).transform('mean')
    # df['cgpa_avg_uni_rating'] = df['cgpa'].groupby(df['university_rating']).transform('mean')
    return df

# Feature names
ordered_features = ['gre_score','toefl_score','cgpa','sop','lor','university_rating','research']