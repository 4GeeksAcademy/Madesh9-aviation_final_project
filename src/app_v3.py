import configuration as config 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import base64

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath("app_v1.py")))
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# PAGE CONFIG & BANNER IMAGE
# ---------------------------
# Set page config
st.set_page_config(
    page_title="Flight Incident Predictor",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------
# LOAD MODEL AND DATA
# ---------------------------
# Load the trained model
model_path = os.path.join(os.path.dirname(base_dir), "models", "model.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)
        
print("Model loaded successfully.")

#load data
route_frequency_path = os.path.join(base_dir, "static", "route_frequency.json")
destination_path = os.path.join(base_dir, "static", "unique_destination.json")
origin_path = os.path.join(base_dir, "static", "unique_origin.json")

with open(origin_path, 'r') as f:
    unique_origin = json.load(f)

with open(destination_path, 'r') as f:
    unique_destination = json.load(f)

with open(route_frequency_path, 'r') as f:
    route_frequency = json.load(f)

# ---------------------------
# FUNCTIONS
# ---------------------------
def check_route(df, route_frequency):
    '''Checks if route in frequency, returns None if not present.'''

    # create and encode route
    df['route'] = df['origin'] + '_' + df['destination']

    if df['route'].iloc[0] not in route_frequency:
        return None
    else:
        df['route_encoded'] = df['route'].map(route_frequency)
        df['route_encoded'].fillna(0, inplace=True)
        df.drop(columns=['route'], inplace=True)

    return df

# create and encode time-sin and time-cosine
def hhmm_to_minutes(hhmm):
    hours, minutes = map(int, hhmm.split(":"))
    return hours * 60 + minutes  

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
# ---------------------------
# MAIN APP
# ---------------------------
def main():    
    
    image_path = "/workspaces/Madesh9-aviation_final_project/src/static/photo.jpg"  # Replace with your local image path
    base64_image = get_base64_image(image_path)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Create two columns for parallel display
    col1, col2, col3 = st.columns([1,2,1])

    # Display GIFs in respective columns
    with col1:
        st.image(os.path.join(os.path.dirname(__file__), "static", "take_off_1.gif"), use_container_width=True)
    
    # Set title and subtitle
    with col2:
        # Add custom CSS with more styling options    
        st.markdown("""
            <style>
            .centered-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 1rem;
            padding: 1rem;
            border-bottom: 2px solid #1E88E5;
            }
            </style>
            """, unsafe_allow_html=True)
        # Use the styled title
        st.markdown('<h1 class="centered-title">Flight Incident Risk Predictor</h1>', unsafe_allow_html=True) 
        
        # Use the styled title
        st.markdown('<p class="centered-text">This app predicts the likelihood of a flight incident based on origin, destination, and departure time information!</p>', unsafe_allow_html=True)
       
    # Display GIFs in respective columns
    with col3:
        st.image(os.path.join(os.path.dirname(__file__), "static", "crash_1.gif"), use_container_width=True)
    st.divider()

    # ---------------------------
    # TABS
    # ---------------------------
    # Custom CSS for tab styling

    tab1, tab2, tab3 = st.tabs(["🔮 Incident Predictor", "📊 Model Performance", "📈 Data Exploration"])
    # TAB 1: Incident Predictor UI Skeleton
    with tab1:
        st.header("Model Performance Visualizations")
    # TAB 2: Model Performance UI Skeleton
    with tab2:
        st.header("Model Performance Visualizations")
        # Create two columns for parallel display
        
            
    # TAB 3: Data oration UI Skeleton
    with tab3:
        st.header("Data Visualizations")   
        # Create two columns for parallel display
        
if __name__ == "__main__":
    main()