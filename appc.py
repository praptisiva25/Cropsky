import streamlit as st
import numpy as np
import joblib  # For loading the trained model and scaler

# Load the trained model, scaler, and label encoder
model = joblib.load('crop_pred_97.pkl')
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

    # Scale the input features using the same scaler used during training
    user_input_scaled = scaler.transform(user_input)

    # Predict using the trained model
    encoded_prediction = model.predict(user_input_scaled)

    # Decode the prediction to get the original label (crop name)
    prediction_label = label_encoder.inverse_transform(encoded_prediction.astype(int))

    return prediction_label[0]


def main():
    st.title('Crop Recommendation System')

    # Collect user inputs without min and max constraints
    N = st.number_input("Enter the value for Nitrogen (N) [0-140]:", value=0.0, step=0.1)
    P = st.number_input("Enter the value for Phosphorus (P) [5-145]:", value=0.0, step=0.1)
    K = st.number_input("Enter the value for Potassium (K) [5-205]:", value=0.0, step=0.1)
    temperature = st.number_input("Enter the temperature in Celsius [8-44]:", value=0.0, step=0.1)
    humidity = st.number_input("Enter the humidity percentage [14-100]:", value=0.0, step=0.1)
    ph = st.number_input("Enter the soil pH value [3-10]:", value=0.0, step=0.1)
    rainfall = st.number_input("Enter the rainfall in mm [20-300]:", value=0.0, step=0.1)

    # Validate inputs
    all_inputs_valid = (
            0 <= N <= 140 and
            5 <= P <= 145 and
            5 <= K <= 205 and
            8 <= temperature <= 44 and
            14 <= humidity <= 100 and
            3 <= ph <= 10 and
            20 <= rainfall <= 300
    )

    # Disable the Predict button if any input is invalid
    if not all_inputs_valid:
        st.warning("Please ensure all inputs are within the correct range.")

    # Predict button is disabled until all inputs are valid
    predict_button = st.button('Predict', disabled=not all_inputs_valid)

    if predict_button:
        # Manual validation and error handling after all inputs are collected
        if not (0 <= N <= 140):
            st.error(f"Nitrogen (N) must be between 0 and 140. You entered: {N}")
        elif not (5 <= P <= 145):
            st.error(f"Phosphorus (P) must be between 5 and 145. You entered: {P}")
        elif not (5 <= K <= 205):
            st.error(f"Potassium (K) must be between 5 and 205. You entered: {K}")
        elif not (8 <= temperature <= 44):
            st.error(f"Temperature must be between 8 and 44 Celsius. You entered: {temperature}")
        elif not (14 <= humidity <= 100):
            st.error(f"Humidity must be between 14 and 100%. You entered: {humidity}")
        elif not (3 <= ph <= 10):
            st.error(f"Soil pH must be between 3 and 10. You entered: {ph}")
        elif not (20 <= rainfall <= 300):
            st.error(f"Rainfall must be between 20 and 300 mm. You entered: {rainfall}")
        else:
            # All inputs are valid, perform prediction
            try:
                recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                st.success(f"The recommended crop for the given conditions is: {recommended_crop}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


if __name__ == '__main__':
    main()
