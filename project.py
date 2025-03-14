streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Ensure matplotlib is imported
from sklearn.preprocessing import LabelEncoder  # Ensure sklearn is available

# Load the trained model
try:
    with open("C:\Users\lenovo\Downloads\flightfare.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'flightfare.pkl' is in the same directory.")
    st.stop()

st.title("Flight Fare Prediction")
st.write("Enter flight details to predict the fare")

# Input fields
airline = st.selectbox("Select Airline", ["Indigo", "Air India", "SpiceJet"])
stops = st.slider("Number of Stops", 0, 3, 1)
duration = st.number_input("Duration (minutes)", min_value=0, value=60)
day_of_week = st.slider("Day of the Week", 1, 7, 1)

# Convert categorical input to numerical (Example: Airline encoding)
airline_mapping = {"Indigo": 0, "Air India": 1, "SpiceJet": 2}
airline_encoded = airline_mapping.get(airline, -1)

if st.button("Predict Fare"):
    try:
        features = np.array([airline_encoded, stops, duration, day_of_week]).reshape(1, -1)
        prediction = model.predict(features)
        st.success(f"Predicted Fare: ₹{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")

# Data Preprocessing Function
def preprocess_data(data):
    try:
        if "Date_of_Journey" in data.columns:
            data["Journey_date"] = pd.to_datetime(data["Date_of_Journey"], format='%d/%m/%Y', errors='coerce').dt.day
            data["Journey_month"] = pd.to_datetime(data["Date_of_Journey"], format='%d/%m/%Y', errors='coerce').dt.month
            data.drop(["Date_of_Journey"], axis=1, inplace=True)

        # Convert Duration to minutes safely
        def convert_duration(duration):
            try:
                parts = duration.split()
                hours = int(parts[0].replace('h', '')) if 'h' in parts[0] else 0
                minutes = int(parts[1].replace('m', '')) if len(parts) > 1 else 0
                return hours * 60 + minutes
            except Exception:
                return 0  # Handle errors gracefully

        if "Duration" in data.columns:
            data["Duration"] = data["Duration"].apply(convert_duration)

        data.drop(["Route", "Additional_Info"], axis=1, inplace=True, errors='ignore')

        label_columns = ["Airline", "Source", "Destination"]
        for col in label_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

        return data
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None

# Example Usage
st.write("\nUpload Flight Data for Prediction:")
uploaded_file = st.file_uploader(""C:\Users\lenovo\Downloads\Data_Train.csv"", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        if df is not None and not df.empty:
            predictions = model.predict(df)
            df["Predicted Fare"] = predictions
            st.write(df)
            st.download_button(label="Download Predictions", data=df.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', mime='text/csv')
        else:
            st.error("Uploaded file contains invalid or empty data.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
