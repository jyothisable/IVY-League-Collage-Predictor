import streamlit as st

import numpy as np
import pandas as pds
import joblib

lr_model = joblib.load('models/LR_model.joblib')


st.title('IVY League Collage Predictor')
