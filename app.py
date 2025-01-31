import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# Connect to SQLite database
conn = sqlite3.connect('form_2.db', check_same_thread=False, timeout=10)
cursor = conn.cursor()

# Create tables for user registration and blockchain data if they don't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS registrations (
        username TEXT(50),
        password TEXT(50),
        surname TEXT(50),
        name TEXT(50),
        age TEXT(50),
        country TEXT(50),
        birthcountry TEXT(50)
    )
    """
)

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS blockchain_data (
        block_index INTEGER,
        block_hash TEXT,
        block_data TEXT
    )
    """
)

conn.commit()

# Link the external CSS file
def load_css():
    st.markdown("""<link rel="stylesheet" href="appCSS.css">""", unsafe_allow_html=True)

load_css()

# Blockchain Implementation
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_index = self.get_last_block_index()  # Start with the last known block index
        self.create_block(0, "0")  # Create the genesis block

    def get_last_block_index(self):
        cursor.execute("SELECT MAX(block_index) FROM blockchain_data")
        result = cursor.fetchone()
        if result[0] is not None:
            return result[0]
        return 0  # Start from 0 if there are no blocks in the database

    def create_block(self, index, previous_hash):
        block = Block(index, time(), {}, previous_hash)
        self.chain.append(block)
        return block

    def add_patient_data(self, patient_data):
        # Increment the current block index
        self.current_index += 1
        last_block = self.chain[-1]
        index = self.current_index
        previous_hash = last_block.hash
        block = self.create_block(index, previous_hash)
        block.data = patient_data  # Add patient data to the new block

        # Save block data to the database
        cursor.execute(
            "INSERT INTO blockchain_data (block_index, block_hash, block_data) VALUES (?, ?, ?)",
            (block.index, block.hash, json.dumps(block.data))
        )
        conn.commit()

        return block

# Initialize blockchain
blockchain = Blockchain()

# Hash password function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup function
def signup():
    st.markdown("""<style>.custom-title {text-align: center;font-size: 20px;font-weight: bold;margin-bottom: 15px;}</style><h1 class="custom-title">Create a New Account</h1>""", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Signup"):
        cursor.execute("INSERT INTO registrations (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        st.success("Account created successfully! Please log in.")

# Login function
def login():
    st.markdown("""<style>.custom-title {text-align: center;font-size: 20px;font-weight: bold;margin-bottom: 15px;}</style><h1 class="custom-title">Login to Your Account</h1>""", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute("SELECT * FROM registrations WHERE username = ? AND password = ?", 
                       (username, hash_password(password)))
        user = cursor.fetchone()
        if user:
            st.session_state.authenticated = True
            st.success(f"Welcome !")
        else:
            st.error("Invalid credentials")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        st.write("You are logged in!")
    else:
        st.write("Please log in.")

# Load and preprocess data for disease prediction
@st.cache_data
def load_data():
    data = pd.read_csv(r"Disease_symptom_and_patient_profile_dataset.csv")
    return data

# Preprocess the data for model training
def preprocess_data(data):
    label_encoder = LabelEncoder()
    binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Outcome Variable']
    for column in binary_columns:
        data[column] = label_encoder.fit_transform(data[column])
    data = pd.get_dummies(data, columns=['Blood Pressure', 'Cholesterol Level'], drop_first=True)
    return data

# Train the model
def train_model():
    data = load_data()
    processed_data = preprocess_data(data)

    X = processed_data.drop(columns=['Disease'])
    y = processed_data['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train

# Create model instance
model, X_train = train_model()

# Patient Details Form
def patient_details_form():
    st.subheader("Patient Details Form")
    with st.container():
        name = st.text_input("Enter your Name")
        age = st.slider("Enter your Age", 1, 100, 30)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        address = st.text_area("Enter your Address")
    return {"Name": name, "Age": age, "Gender": gender, "Address": address}

# Symptom Checker Form
def user_input_features():
    st.subheader("Symptom Checker")
    col1, col2 = st.columns(2)

    with col1:
        Fever = st.selectbox('Fever', ('Yes', 'No'))
        Cough = st.selectbox('Cough', ('Yes', 'No'))
        Fatigue = st.selectbox('Fatigue', ('Yes', 'No'))
        Difficulty_Breathing = st.selectbox('Difficulty Breathing', ('Yes', 'No'))

    with col2:
        Blood_Pressure = st.selectbox('Blood Pressure', ('Low', 'Normal', 'High'))
        Cholesterol_Level = st.selectbox('Cholesterol Level', ('Low', 'Normal', 'High'))

    data = {
        'Fever': 1 if Fever == 'Yes' else 0,
        'Cough': 1 if Cough == 'Yes' else 0,
        'Fatigue': 1 if Fatigue == 'Yes' else 0,
        'Difficulty Breathing': 1 if Difficulty_Breathing == 'Yes' else 0,
        'Blood Pressure_Low': 1 if Blood_Pressure == 'Low' else 0,
        'Blood Pressure_Normal': 1 if Blood_Pressure == 'Normal' else 0,
        'Cholesterol Level_Low': 1 if Cholesterol_Level == 'Low' else 0,
        'Cholesterol Level_Normal': 1 if Cholesterol_Level == 'Normal' else 0
    }

    features = pd.DataFrame(data, index=[0])
    missing_cols = set(X_train.columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    features = features[X_train.columns]

    return features

# Disease Prediction
def predict_disease(features):
    prediction = model.predict(features)[0]
    return prediction

# Main App Logic
def app():
    # Ensure the session state has an 'authenticated' key
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        st.image('Assest\Doctor Pic.png', use_container_width=True)

    st.markdown("""<style>.custom-title {text-align: center;font-size: 50px;font-weight: bold;margin-bottom: 20px;}</style><h1 class="custom-title">Welcome to AI Doctor</h1>""", unsafe_allow_html=True)

    # If the user is not authenticated, show login or signup option
    if not st.session_state.authenticated:
        st.markdown("""<style>.custom-title {text-align: center;font-size: 20px;font-weight: bold;margin-bottom: 20px;}</style><h1 class="custom-title">Please log in or sign up to continue.</h1>""", unsafe_allow_html=True)
        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        if choice == 'Sign up':
            signup()
        elif choice == 'Login':
            login()

    # If the user is authenticated, show the "Log out" button and other content
    if st.session_state.authenticated:
        # Show the "Log out" button
        if st.button("Log out"):
            st.session_state.authenticated = False  # Log out the user
            st.success("You have been logged out.")
            st.rerun()  # Refresh the app to update the state

        st.markdown("""<style>.custom-title {text-align: center;font-size: 20px;font-weight: bold;margin-bottom: 20px;}</style><h1 class="custom-title">Please enter your details and symptoms to get a prediction:</h1>""", unsafe_allow_html=True)

        # Centered patient details form
        patient_info = patient_details_form()

        # Symptom checker form
        features = user_input_features()

        if st.button("Submit"):
            st.subheader("Patient Details Submitted:")
            st.markdown(f"""
            <div class="patient-details">
                <div><h4>Name: {patient_info['Name']}</h4></div>
                <div><h4>Age: {patient_info['Age']}</h4></div>
                <div><h4>Gender: {patient_info['Gender']}</h4></div>
                <div><h4>Address: {patient_info['Address']}</h4></div>
            </div>
            """, unsafe_allow_html=True)

            # Predict Disease
            prediction = predict_disease(features)
            st.subheader("Predicted Disease:")
            st.write(f"<h2 style='text-align: center; color: #4e73df;'>{prediction}</h2>", unsafe_allow_html=True)

            # Add patient data to blockchain
            blockchain.add_patient_data(patient_info)

if __name__ == "__main__":
    app()
