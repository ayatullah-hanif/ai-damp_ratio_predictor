#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("concrete_damping_dataset.csv")


# In[3]:


# Drop missing values
df.dropna(inplace=True)

# Features and target
features = ['rubber_content', 'aggregate_size', 'water_cement_ratio', 'compressive_strength', 'flexural_strength']
target = 'damping_ratio'

X = df[features]
y = df[target]


# In[5]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "damping_model.pkl")
print("✅ Model trained and saved as 'damping_model.pkl'")


# **APP UI**

# In[8]:


import streamlit as st
import joblib
import pandas as pd

# Streamlit setup
st.set_page_config(page_title="Damping Ratio Predictor", layout="wide")

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("damping_model.pkl")
except:
    st.error("⚠️ Model file not found. Please run train_model.py first.")
    st.stop()

# App UI
st.markdown("""
<div class="bg-gray-100 p-6 rounded-lg shadow-lg max-w-3xl mx-auto">
    <h1 class="text-3xl font-bold text-center text-gray-800 mb-4">Damping Ratio Predictor</h1>
    <p class="text-gray-600 text-center mb-6">Enter concrete mix properties to estimate the damping ratio.</p>
""", unsafe_allow_html=True)

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        rubber_content = st.number_input("Rubber Content (%)", 0.0, 100.0, 5.0)
        aggregate_size = st.number_input("Aggregate Size (mm)", 2.0, 40.0, 10.0)
        water_cement_ratio = st.number_input("Water-Cement Ratio", 0.2, 1.0, 0.5)

    with col2:
        compressive_strength = st.number_input("Compressive Strength (MPa)", 5.0, 100.0, 30.0)
        flexural_strength = st.number_input("Flexural Strength (MPa)", 1.0, 20.0, 5.0)

    submitted = st.form_submit_button("Predict Damping Ratio")

# Make prediction
if submitted:
    user_input = pd.DataFrame([{
        'rubber_content': rubber_content,
        'aggregate_size': aggregate_size,
        'water_cement_ratio': water_cement_ratio,
        'compressive_strength': compressive_strength,
        'flexural_strength': flexural_strength
    }])

    prediction = model.predict(user_input)[0]

    st.markdown(f"""
    <div class="mt-6 p-4 bg-white rounded-lg shadow-md">
        <h2 class="text-2xl font-semibold text-gray-800 mb-4">Prediction Result</h2>
        <p class="text-gray-700 text-lg">🔍 Predicted Damping Ratio: <span class="text-blue-600 font-bold">{prediction:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




