import streamlit as st
import pickle
import numpy as np
import json

# Load model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# UI
st.title("🏠 House Price Prediction System")

# Inputs
location = st.selectbox("Location", data_columns[3:])
sqft = st.number_input("Total Sqft", value=1000)
bath = st.number_input("Bathrooms", value=2)
bhk = st.number_input("BHK", value=2)

# Prediction function
def predict_price(location, sqft, bath, bhk):
    loc_index = data_columns.index(location)

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Button
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Estimated Price: ₹ {round(price,2)} Lakhs")