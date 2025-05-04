import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# Load the PMML model
pmml_model = Model.load('gbm_model.pmml')
# Load the data
dev = pd.read_csv('dev_finally.csv')
vad = pd.read_csv('vad_finally.csv')

# Define feature names in the correct order (from PMML model)
feature_names = ['smoker', 'drink','sleep','Hypertension','HRR', 'NLR','LMR',
                 'INDFMPIR',  'LBXWBCSI', 'LBXRBCSI','LBXPLTSI']

# Streamlit user interface
st.title("Co-occurrence of Myocardial Infarction and Stroke Predictor")

# Create input columns to organize widgets better
col1, col2 = st.columns(2)

with col1:
    smoker = st.selectbox("Smoker:", options=[1, 2, 3], 
                         format_func=lambda x: "Never" if x == 1 else "Former" if x == 2 else "Current")
    drink = st.selectbox("Alcohol Consumption:", options=[1, 2], 
                        format_func=lambda x: "No" if x == 2 else "Yes")
    sleep = st.selectbox("Sleep Problem:", options=[1, 2], 
                         format_func=lambda x: "Yes" if x == 1 else "No")
    Hypertension = st.selectbox("Hypertension:", options=[1, 2], 
                                format_func=lambda x: "No" if x == 2 else "Yes")
    HRR = st.number_input("HRR Ratio:", min_value=0.23, max_value=5.0, value=0.62)
    NLR = st.number_input("NLR Ratio:", min_value=0.01, max_value=10.0, value=5.22)
    
with col2:
    LMR = st.number_input("LMR Ratio:", min_value=0.01, max_value=20.0, value=5.8)
    INDFMPIR = st.number_input("Poverty Income Ratio:", min_value=0.1, max_value=5.0, value=0.9)
    LBXWBCSI = st.number_input("White Blood Cell Count (10^3/μL):", min_value=1.0, max_value=200.0, value=9.4)
    LBXRBCSI = st.number_input("Red Blood Cell Count (10^6/μL):", min_value=1.0, max_value=10.0, value=4.06)
    LBXPLTSI = st.number_input("Platelet Cell Count (10^3/μL):", min_value=1.0, max_value=1000.0, value=253.0)

# Process inputs and make predictions
feature_values = [smoker, drink,sleep,Hypertension,HRR, NLR,LMR,
                 INDFMPIR,  LBXWBCSI, LBXRBCSI,LBXPLTSI]

if st.button("Predict"):
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Make prediction
    prediction = pmml_model.predict(input_df)
    prob_0 = prediction['probability(1)'][0]
    prob_1 = prediction['probability(0)'][0]
    
    # Determine predicted class
    predicted_class = 1 if prob_1 > 0.436018256400085 else 0
    probability = prob_1 if predicted_class == 1 else prob_0
    
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Comorbidity, 0: Non-comorbidity)")
    st.write(f"**Probability of Comorbidity:** {prob_1:.4f}")
    st.write(f"**Probability of Non-comorbidity:** {prob_0:.4f}")

    # Generate advice
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of co-occurrence of myocardial infarction and stroke disease. "
            f"The model predicts a {probability*100:.1f}% probability. "
            "It's advised to consult with your healthcare provider for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk ({(1-probability)*100:.1f}% probability). "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups."
        )
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Explanation")
    
    # Prepare background data (using first 100 samples)
    background = vad[feature_names].iloc[:100]
    
    # Define prediction function for SHAP
    def pmml_predict(data):
        if isinstance(data, pd.DataFrame):
            input_df = data[feature_names].copy()
        else:
            input_df = pd.DataFrame(data, columns=feature_names)
        
        predictions = pmml_model.predict(input_df)
        return np.column_stack((predictions['probability(0)'], predictions['probability(1)']))
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(pmml_predict, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_df)
    
    # Display SHAP force plot
    st.subheader("SHAP Force Plot Explanation")
    plt.figure()
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[0], 
                       shap_values[0,:,0],  # Take SHAP values for class 1
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
    else:
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0,:,1],  # Take SHAP values for class 0
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
    
    st.pyplot(plt.gcf())
    plt.clf()

   # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=background.values,
        feature_names=feature_names,
        class_names=['Non-comorbidity', 'Comorbidity'],
        mode='classification'
    )
    
    lime_exp = lime_explainer.explain_instance(
      data_row=input_df.values.flatten(),
      predict_fn=lambda x: 1-pmml_predict(x)  # 反转概率
    )
    
    # Display LIME explanation
    lime_html = lime_exp.as_html(show_table=False)  
    st.components.v1.html(lime_html, height=800, scrolling=True)
