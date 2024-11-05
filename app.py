# Import necessary libraries for the app
import streamlit as st  # Streamlit is a library used to create interactive web apps for data science
import pandas as pd  # Pandas is used for data manipulation and handling tabular data
import numpy as np  # Numpy is a library for numerical operations, especially on arrays and matrices
import psycopg2  # Psycopg2 is used to connect Python applications to a PostgreSQL database
import joblib  # Joblib is used to save and load machine learning models

# This section sets up the Streamlit page configuration
st.set_page_config(page_title="Rental Property Price Prediction", layout="wide")

# Title of the Streamlit app
st.title("Rental Property Price Prediction App")

# Function to load the machine learning model from a file
def load_model():
    """
    This function loads the trained machine learning model
    from a file using joblib. 
    """
    return joblib.load("model.pkl")  # 'model.pkl' is the file where the ML model is saved

# Load the model by calling the function
model = load_model()

# Function to connect to the PostgreSQL database
def connect_to_db():
    """
    This function connects to the PostgreSQL database
    using psycopg2 and returns the connection and cursor objects.
    """
    try:
        conn = psycopg2.connect(
            dbname="rental_db",  # Database name
            user="postgres",     # Database user
            password="your_password",  # Password (replace with actual password)
            host="localhost",    # Server address, localhost means local machine
            port="5432"          # PostgreSQL default port
        )
        return conn, conn.cursor()
    except Exception as e:
        st.error("Error connecting to the database: " + str(e))
        return None, None

# Function to fetch property data from the database
def fetch_data_from_db():
    """
    Fetches property data from the PostgreSQL database
    for use in the app. Assumes 'properties' table is available.
    """
    conn, cursor = connect_to_db()
    if conn is None:
        return None
    query = "SELECT * FROM properties;"
    cursor.execute(query)
    data = cursor.fetchall()
    # Convert fetched data to a DataFrame for easy manipulation
    df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
    conn.close()
    return df

# Load data from the database
data = fetch_data_from_db()

# Display data in the app, if available
if data is not None:
    st.subheader("Available Properties Data")
    st.write(data)  # Shows the data in a table format in the Streamlit app

# Sidebar for user input to predict rental prices
st.sidebar.header("Input Parameters for Prediction")

# Function to capture user input for prediction
def user_input_features():
    """
    Captures input values from the user for prediction.
    Returns the inputs as a dictionary.
    """
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 1)
    sqft = st.sidebar.slider("Square Footage", 500, 5000, 1000)
    locality = st.sidebar.selectbox("Locality", options=["Area1", "Area2", "Area3"])  # Example areas
    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "locality": locality
    }

# Get the user's inputs
input_data = user_input_features()

# Convert inputs into a format suitable for the model (DataFrame)
input_df = pd.DataFrame([input_data])

# Predict rental price based on user inputs using the loaded model
def predict_rent(model, input_df):
    """
    Uses the ML model to predict the rental price based on input data.
    """
    try:
        prediction = model.predict(input_df)  # Predicts rent for given input
        return prediction[0]  # Return the first (and only) prediction
    except Exception as e:
        st.error("Error in making prediction: " + str(e))
        return None

# Show the prediction if model and inputs are valid
predicted_price = predict_rent(model, input_df)
if predicted_price is not None:
    st.header("Predicted Rental Price")
    st.write(f"${predicted_price:,.2f}")  # Displays prediction formatted as currency
