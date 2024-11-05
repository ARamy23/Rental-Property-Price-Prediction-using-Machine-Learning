# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import joblib
from sklearn.ensemble import RandomForestRegressor  # Using a Random Forest for training
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit page configuration
st.set_page_config(page_title="Rental Property Price Prediction", layout="wide")
st.title("Rental Property Price Prediction App")

# Function to train the model (if `model.pkl` is missing)
def train_model():
    """
    This function trains a Random Forest model using sample data,
    then saves it as 'model.pkl' for future use.
    """
    st.write("Training model as 'model.pkl' not found...")
    
    # Sample data for demonstration (replace with real dataset if available)
    data = {
        "bedrooms": [1, 2, 3, 2, 3],
        "bathrooms": [1, 2, 2, 1, 3],
        "sqft": [500, 1500, 2000, 1200, 1800],
        "locality": [1, 2, 2, 1, 3],  # Encoded locality for simplicity
        "rent": [1000, 2000, 2500, 1800, 2700]
    }
    df = pd.DataFrame(data)
    X = df.drop("rent", axis=1)
    y = df["rent"]
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training a Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")  # Save the model
    st.write("Model training complete and saved as 'model.pkl'.")
    
    # Check model accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model trained with Mean Squared Error: {mse:.2f}")

# Function to load the model
def load_model():
    """
    Loads the trained machine learning model from a file.
    If 'model.pkl' is missing, trains a new model.
    """
    try:
        model = joblib.load("model.pkl")
        st.write("Loaded model successfully.")
        return model
    except FileNotFoundError:
        st.warning("Model file 'model.pkl' not found. Training a new model...")
        train_model()  # Train the model if not found
        return joblib.load("model.pkl")

# Load the model
model = load_model()

# Function to connect to the PostgreSQL database
def connect_to_db():
    """
    Connects to the PostgreSQL database using psycopg2.
    """
    try:
        st.write("Connecting to the database...")
        conn = psycopg2.connect(
            dbname="rental_db",
            user="postgres",
            password="your_password",  # Replace with the actual password
            host="localhost",
            port="5432"
        )
        st.write("Connected to the database successfully.")
        return conn, conn.cursor()
    except Exception as e:
        st.error("Failed to connect to the database: " + str(e))
        return None, None

# Function to fetch data from the database
def fetch_data_from_db():
    """
    Fetches property data from the database.
    """
    conn, cursor = connect_to_db()
    if conn is None:
        st.error("Database connection failed.")
        return None
    
    st.write("Fetching data from the database...")
    query = "SELECT * FROM properties;"
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        st.write("Data fetched successfully.")
        # Convert fetched data to DataFrame
        df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
        conn.close()
        return df
    except Exception as e:
        st.error("Error fetching data: " + str(e))
        return None

# Load and display data
data = fetch_data_from_db()
if data is not None:
    st.subheader("Available Properties Data")
    st.write(data)

# Sidebar for input parameters
st.sidebar.header("Input Parameters for Prediction")

# Function to capture user inputs
def user_input_features():
    """
    Captures user inputs for prediction.
    """
    st.sidebar.write("Choose the features for prediction.")
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 1)
    sqft = st.sidebar.slider("Square Footage", 500, 5000, 1000)
    locality = st.sidebar.selectbox("Locality", options=["Area1", "Area2", "Area3"])
    locality_encoded = ["Area1", "Area2", "Area3"].index(locality) + 1
    return pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "locality": locality_encoded
    }])

# Get user inputs and display
input_data = user_input_features()
st.write("User input for prediction:", input_data)

# Prediction function
def predict_rent(model, input_df):
    """
    Predicts the rental price based on user inputs.
    """
    st.write("Running model prediction...")
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Rental Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error("Prediction failed: " + str(e))

# Make prediction
predict_rent(model, input_data)
