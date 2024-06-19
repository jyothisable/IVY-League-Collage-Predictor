import streamlit as st

import numpy as np
import pandas as pd
import joblib

from build_features import * # import all functions in build_features.py
# #column transformer

lr_model = joblib.load('models/LR_model.joblib')


st.title('IVY League Collage Predictor :medal:')

# Add a banner image
st.image('banner.jpg', use_column_width=True)

st.write("**Enter your details to predict the chance of admission to IVY league college**")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Define the input fields
    cgpa = st.number_input("CGPA", min_value=1.0, max_value=10.0, step=0.1, value=8.0)
    gre_score = st.number_input("GRE Score", min_value=250, max_value=340, step=1, value=300)
    toefl_score = st.number_input("TOEFL Score", min_value=50, max_value=120, step=1, value=105)
    
with col2:
    university_rating = st.selectbox('University Rating', options=[1, 2, 3, 4, 5])
    research = st.checkbox("Research Experience")



# Create a button to submit the form
if st.button("Submit",use_container_width=10):
    # Create a dictionary with the input values
    data = {
        "gre_score": gre_score,
        "toefl_score": toefl_score,
        "cgpa": cgpa,
        "sop":3.0,
        "lor":3.0,
        "university_rating": university_rating,
        "research": int(research)  # Convert boolean to integer
    }
    

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([data])
    # st.write(input_data)
    
    chance_of_admit = lr_model.predict(input_data)[0]
    
    
    chance_of_admit = min(1,chance_of_admit) # limit the chance of admission to 1
    st.header(f'Chance of Admission: {chance_of_admit*100:.2f}%')
    
