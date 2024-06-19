
# Scientific libraries
import numpy as np
import pandas as pd

# Logging
import logging

# Visual libraries
import matplotlib.pyplot as plt
import seaborn as sns

# To save model for deployment
import joblib

# Helper libraries
import urllib.request
from tqdm.notebook import tqdm, trange # Progress bar
#from colorama import Fore, Back, Style # coloured text in output


# Visual setup
# %config InlineBackend.figure_format = 'retina' # sets the figure format to 'retina' for high-resolution displays.

# display estimators as diagrams
from sklearn import set_config
set_config(display='diagram')


# Pandas options
from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all' # display all interaction 

# Table styles
table_styles = {
    'cerulean_palette': [
        dict(selector="th", props=[("color", "#FFFFFF"), ("background", "#004D80")]),
        dict(selector="td", props=[("color", "#333333")]),
        dict(selector="table", props=[("font-family", 'Arial'), ("border-collapse", "collapse")]),
        dict(selector='tr:nth-child(even)', props=[('background', '#D3EEFF')]),
        dict(selector='tr:nth-child(odd)', props=[('background', '#FFFFFF')]),
        dict(selector="th", props=[("border", "1px solid #0070BA")]),
        dict(selector="td", props=[("border", "1px solid #0070BA")]),
        dict(selector="tr:hover", props=[("background", "#80D0FF")]),
        dict(selector="tr", props=[("transition", "background 0.5s ease")]),
        dict(selector="th:hover", props=[("font-size", "1.07rem")]),
        dict(selector="th", props=[("transition", "font-size 0.5s ease-in-out")]),
        dict(selector="td:hover", props=[('font-size', '1.07rem'),('font-weight', 'bold')]),
        dict(selector="td", props=[("transition", "font-size 0.5s ease-in-out")])
    ]
}

#from rich import print # color from print statement 
# Seed value for numpy.random => makes notebooks stable across runs
np.random.seed(42)

# Filter warnings (to be done after everything is done)
import warnings
warnings.filterwarnings('ignore')

class DataIngestor:
    """"
    A class to handle downloading data and loading it into a pandas dataframe along with basic sanity options
    """
    def __init__(self, input_path : str = '../data/raw/', output_path : str = '../data/processed/'):
        self.input_path = input_path
        self.output_path = output_path
        self.file_path = None
        self.url = None
    
    def download_data(self, url : str = None) -> None:
        self.url = url
        logging.info(f'Downloading data from {self.url}')
        # set file path from url
        self.file_path = f"{self.input_path}{url.split('/')[-1]}" 
        urllib.request.urlretrieve(self.url, self.file_path)
        return self.file_path

    def load_data(self, file_name : str = None) -> pd.DataFrame:
        # explicitly given file name => load from there otherwise use url download path (previous path)
        self.file_path= f"{self.input_path}{file_name}" if file_name is not None else self.file_path
        logging.info(f'Ingesting data from {self.file_path}')
        #TODO add csv check
        return pd.read_csv(self.file_path)
    
    def save_data(self, df : pd.DataFrame, file_name : str) -> None:
        logging.info(f'Saving data to {self.output_path}')
        df.to_csv(self.output_path+file_name)
        
    def sanitize(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to snake case and strip whitespace
        """
        return df.rename(lambda x: x.lower().strip().replace(' ', '_'),axis='columns')
    
data_url = DataIngestor()

# data_url.download_data(url = 'https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/839/original/Jamboree_Admission.csv')
# data_url.file_path

df = data_url.sanitize(data_url.load_data('Jamboree_Admission.csv'))

class DataWrangler:
    """"
    A class to handle cleaning and wrangling data (my data cleaning specialist)
    """
    def __init__(self, df : pd.DataFrame):
        self.df = df
        self.processed_df = pd.DataFrame()
        
        # Range constrain attributes
        self.range_constrains = {}
        self.failed_index_range_constrains = pd.Index([])
        
        # Data type constrain attributes
        self.data_type_map = {}
        
        # Unique constrain attributes
        self.unique_cols = []
        self.failed_unique_cols = pd.Index([])
        
    
    def set_data_type(self, data_type_map : dict) -> pd.DataFrame:
        self.data_type_map = data_type_map
        self.processed_df = self.df.astype(data_type_map)
        return self.processed_df
    
    def check_range_constrain(self, constrains : dict) -> bool:
        self.range_constrains = constrains
        mask = np.array([False]*len(self.df))
        for col,(min_range, max_range) in self.range_constrains.items(): 
            # Adding mask => we select the row if at least one entry is out of range in that row. (Cumulative logical OR on array)
            mask += (self.df[col] < min_range) | (self.df[col] > max_range)
        self.failed_index_range_constrains = self.df[mask].index
        if mask.any():
            print('Range constrain failed for below rows:')
            print(self.df.iloc[self.failed_index_range_constrains])
            return False
        else : return True
    
    def fix_range_constrain(self) -> pd.DataFrame:
        logging.info(f'Removing entreis for fixing range constrain for {self.failed_index_range_constrains}')
        self.processed_df = self.processed_df.drop(self.failed_index_range_constrains,errors='ignore')
        return self.processed_df
    
    def check_unique(self, unique_cols : list) -> bool:
        self.unique_cols = unique_cols
        mask = np.array([False]*len(self.df))
        for col in unique_cols:
            mask += self.df[col].duplicated()
        self.failed_unique_cols = self.df[mask].index
        if mask.any():
            print('Unique constrain failed for below rows:')
            print(self.df.iloc[self.failed_unique_cols])
            return False
        else : return True
    
    def fix_unique(self) -> pd.DataFrame:
        # TODO add an option to average the non categorical values based on unique columns using groupby 
        logging.info(f'Removing entreis for fixing unique constrain for {self.failed_unique_cols}')
        self.processed_df = self.processed_df.drop(self.failed_unique_cols,errors='ignore')
        return self.processed_df
    
    def set_categorical_order(self, order : dict) -> pd.DataFrame:
        if self.data_type_map is None:
            raise Exception('Please set data type first')
        self.set_categorical_order = order
        for col, order in self.set_categorical_order.items():
            self.processed_df[col] = self.processed_df[col].cat.set_categories(order, ordered=True)
        return self.processed_df 

# Define range constrains
range_constrains = {
    'gre_score': [0, 340],
    'toefl_score': [0, 120],
    'university_rating': [0, 5],
    'sop': [0, 5],
    'lor': [0, 5],
    'cgpa': [0, 10],
    'research': [0, 1],
    'chance_of_admit': [0, 1]
}

# Define data type
data_type_map = {
    'serial_no.': 'int32',
    'gre_score': 'int32',
    'toefl_score': 'int32',
    'university_rating': 'category',
    'sop': 'category',
    'lor': 'category',
    'cgpa': 'float32',
    'research': 'category',
    'chance_of_admit': 'float32'
}

# Specify unique columns
unique_cols = ['serial_no.']

# Categorical order (this will be inferred by OrdianlEncoder later)
categorical_order = {
    'university_rating': [1, 2, 3, 4, 5],
    'sop': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    'lor': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    'research': [0, 1]
}

# Initialize DataWrangler
clean_data = DataWrangler(df)
clean_data.set_data_type(data_type_map)

# Check range constrains
if not clean_data.check_range_constrain(range_constrains):
    clean_data.fix_range_constrain()

# Check unique constrain
if not clean_data.check_unique(unique_cols):
    clean_data.fix_unique()

# set categorical order
clean_data.set_categorical_order(categorical_order)

# Final data after cleaning
cleaned_df = clean_data.processed_df
data_url.save_data(cleaned_df,'processed.csv')

from sklearn.model_selection import train_test_split
X = cleaned_df.drop('chance_of_admit', axis=1)
y = cleaned_df['chance_of_admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=40, 
                                                    stratify= cleaned_df['sop'])

# Adding an imputation pipeline in case if future datasets have missing values
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

# Specify columns for imputation
num_cols = ['gre_score','toefl_score','cgpa']
cat_cols = ['university_rating','research']

# ct_impute.set_output(transform="pandas")

from sklearn.preprocessing import StandardScaler,OrdinalEncoder

# Column Transformer
from sklearn.compose import make_column_transformer




# Below features are selected after EDA and Diagnostics of VIF (iterative process)
num_cols = ['gre_score','toefl_score','cgpa']
cat_cols = ['research','university_rating']
ordred_feature_name = num_cols + cat_cols # to be used name features back after modeling to know coefficients

ct_feature_scaling = make_column_transformer( # feature scaling
        (   # Numerical Pipeline =>  StandardScaler
            make_pipeline( StandardScaler()), 
            num_cols
        ),
        (   # Categorical Pipeline => OrdinalEncoder
            make_pipeline(OrdinalEncoder()), 
            cat_cols
        )
)

# Make feature Engineering Pipeline
feature_engg_pipeline = make_pipeline(# no imputation needed for new features
    ct_feature_scaling,
    # verbose=True
)
# feature_engg_pipeline.set_output(transform="pandas")

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet,LinearRegression

# Pipeline for target variable
target_pipe = make_pipeline( # Impute + Scale
    SimpleImputer(strategy='mean',add_indicator=True),
    StandardScaler()
)


LR_model = TransformedTargetRegressor(
    regressor=make_pipeline( # Impute -> feature engg (feature gen + feature scaling and encoding) -> Linear Regression
        feature_engg_pipeline, # <- features selected by iterative process like VIF mentioned in Diagnosis section
        LinearRegression()
        # verbose=True
    ),
    transformer=target_pipe, #  impute <=> scale (Takes inverse of target to get back to original scale)
    check_inverse=False
)  

LR_model.fit(X_train,y_train)

joblib.dump(LR_model, '../models/LR_model.joblib')
