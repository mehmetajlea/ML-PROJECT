import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


#Page Configuration
st.set_page_config(page_title="Clothing Review Predictor", layout="wide")

st.title("🛍️ Client Review Prediction System")
st.markdown("Analyze client reviews and predict their recommendation probability.")

#Load Model & Data
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    importance = pd.read_csv('feature_importance.csv')
    return model, importance

model, importance_data = load_model()

#User Input (Sidebar)
st.sidebar.header("Client Data Input")

# Create inputs matching the dataset columns
age = st.sidebar.slider("Client Age", 18, 99, 35)
rating = st.sidebar.slider("Rating Given", 1, 5, 4)
feedback = st.sidebar.number_input("Positive Feedback Count", 0, 100, 2)

# Categorical inputs
division = st.sidebar.selectbox("Division Name", ['General', 'General Petite', 'Intimates'])
dept = st.sidebar.selectbox("Department Name", ['Tops', 'Dresses', 'Bottoms', 'Intimates', 'Jackets', 'Trend'])
cls = st.sidebar.selectbox("Class Name", ['Dresses', 'Knits', 'Blouses', 'Pants', 'Fine gauge', 'Skirts', 'Jackets', 'Lounge', 'Swim', 'Outerwear', 'Shorts', 'Sleep'])

# Text Length (Simulated input)
text_length = st.sidebar.slider("Review Text Length", 0, 1000, 100)

#Create DataFrame for Prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Rating': [rating],
    'Positive Feedback Count': [feedback],
    'Division Name': [division],
    'Department Name': [dept],
    'Class Name': [cls],
    'Text Length': [text_length]
})

# Prediction
if st.button("Predict Recommendation"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display Result
    col1, col2 = st.columns(2)
    col1.metric("Prediction", "Recommended ✅" if prediction == 1 else "Not Recommended ❌")
    col2.metric("Confidence", f"{probability*100:.2f}%")
    
    #Visualization: Risk / Probability Gauge
    st.subheader("Recommendation Probability Distribution")
    fig, ax = plt.subplots()
    colors = ['red', 'green']
    data = [1-probability, probability] # [Not Rec, Rec]
    ax.pie(data, labels=['Not Recommended', 'Recommended'], autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

#Interactive Feature Importance
st.subheader("Feature Importance (Global Model)")
fig_imp, ax_imp = plt.subplots(figsize=(10,6))
sns.barplot(x='abs_coef', y='feature', data=importance_data, hue='feature', palette='viridis', legend=False, ax=ax_imp)
ax_imp.set_title("Top Factors Influencing Prediction")
ax_imp.set_xlabel("Coefficient Strength")
ax_imp.set_ylabel("")
st.pyplot(fig_imp)

# Export Result
st.subheader("Export Results")
if st.button("Download Result as CSV"):
    # Combine input and prediction
    prediction_val = model.predict(input_data)[0]
    prob_val = model.predict_proba(input_data)[0][1]
    input_data['Prediction'] = prediction_val
    input_data['Confidence (%)'] = prob_val * 100
    
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='client_prediction.csv',
        mime='text/csv'
    )