import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  # For loading the trained model and scaler

# Load the trained model, scaler, and label encoder
model = joblib.load('crop_pred_ns2.pkl')
scaler = joblib.load('scaler2.pkl')
label_encoder = joblib.load('label_encoder2.pkl')


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predicts the recommended crop based on the given features.

    Parameters:
    - N: Nitrogen content (float)
    - P: Phosphorus content (float)
    - K: Potassium content (float)
    - temperature: Temperature in Celsius (float)
    - humidity: Humidity percentage (float)
    - ph: Soil pH value (float)
    - rainfall: Rainfall in mm (float)

    Returns:
    - A string representing the recommended crop.
    """
    # Combine inputs into a single array
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale the input features using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict using the trained model
    encoded_prediction = model.predict(user_input_scaled)

    # Decode the prediction to get the original label
    prediction_label = label_encoder.inverse_transform(encoded_prediction.astype(int))

    return prediction_label[0]


def main():
    st.title('Crop Recommendation System')

    # Collect user inputs
    N = st.number_input("Enter the value for Nitrogen (N)", min_value=10.0, max_value=100.0, step=0.1,value=50.0)
    P = st.number_input("Enter the value for Phosphorus (P)", min_value=10.0, max_value=100.0, step=0.1,value=50.0)
    K = st.number_input("Enter the value for Potassium (K)", min_value=10.0, max_value=100.0, step=0.1,value=50.0)
    temperature = st.number_input("Enter the temperature in Celsius", min_value=10.0, max_value=50.0, step=0.1,value=25.0)
    humidity = st.number_input("Enter the humidity percentage", min_value=10.0, max_value=100.0, step=0.1,value=50.0)
    ph = st.number_input("Enter the soil pH value", min_value=1.0, max_value=14.0, step=0.1,value=7.0)
    rainfall = st.number_input("Enter the rainfall in mm", min_value=10.0, max_value=1000.0, step=0.1,value=500.0)

    # When the user clicks the button
    if st.button('Predict'):
        try:
            # Perform prediction
            recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop for the given conditions is: {recommended_crop}")
        except AttributeError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
