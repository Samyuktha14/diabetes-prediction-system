import streamlit as st
import pickle

# Load all models from the pickle file
with open("diabetes_model.pkl", "rb") as file:
    models= pickle.load(file)

# Input fields for user features
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict if you are diabetic:")

age = st.number_input("Age", min_value=0, max_value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0)
glucose_level = st.number_input("Glucose Level", min_value=0.0, max_value=300.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=500.0)
family_history = st.number_input("Family History Score (0 to 1)", min_value=0.0, max_value=1.0)

# Preprocess the input data
input_data = [age, bmi, glucose_level, blood_pressure, insulin, family_history]

# Convert input_data to the expected format for model input (2D array)
input_data = [input_data]

# Prediction button
if st.button("Predict"):
    # Make predictions using all models
    predictions = []
    
    # Loop through models without showing intermediate results
    for model_name, model in models.items():
        try:
            prediction = model.predict(input_data)
            predictions.append(prediction[0])  # Store the result (0 or 1)
        except Exception:
            continue  # If there's an error, just skip that model

    # Calculate the majority prediction
    diabetic_count = sum(predictions)  # Count of 1's (diabetic predictions)
    total_models = len(predictions)

    # Determine final conclusion based on majority voting
    if diabetic_count > total_models / 2:
        final_prediction = "Diabetic"
    else:
        final_prediction = "Non-Diabetic"

    # Display the final conclusion
    st.write(f"Based on the majority of models, the person is **{final_prediction}**.")
