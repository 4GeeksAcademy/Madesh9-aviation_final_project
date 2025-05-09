import configuration as config 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import pydeck as pdk
import time
from PIL import Image

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

def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(url, header=None, names=cols)
    us_airports = df[(df['Country'] == 'United States') & (df['IATA'].notnull()) & (df['IATA'] != '\\N')]
    us_airports["Label"] = us_airports["City"] + " - " + us_airports["Name"] + " (" + us_airports["IATA"] + ")"
    return us_airports[['Label', 'Name', 'City', 'IATA', 'Latitude', 'Longitude']]

# ---------------------------
# ARCS & MAP HELPERS
# ---------------------------
def generate_arc(p1, p2, num_points=100, height=10):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    curve = np.sin(np.linspace(0, np.pi, num_points)) * height
    curved_lats = lats + curve / 100
    return [[lon, lat] for lon, lat in zip(lons, curved_lats)]

icon_data = {
    "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    "width": 128,
    "height": 128,
    "anchorY": 128,
}

def tab4func(airports:pd.DataFrame, model:callable, frequencies:dict):
    '''Defines incident prediction tab.'''

    st.header("Predict Flight Incident Risk")
    col1, col2 = st.columns(2)
    with col1:
        origin_label = st.selectbox("Origin Airport", airports['Label'])
        dest_label = st.selectbox("Destination Airport", airports['Label'])
    with col2:
        departure_time = st.number_input("Departure Time (e.g., 1430 for 2:30 PM)", min_value=0, max_value=2359, step=5)

    if st.button("Predict Incident Probability"):
            st.warning("Prediction logic not implemented.")

    return origin_label, dest_label
# ---------------------------
# MAIN APP
# ---------------------------
def main():    
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔮 Incident Predictor", "📊 Model Performance", "📈 Data Exploration", "🔮 Flight map and animation", "🗺️ Map", "✈️ Animation"])
    # TAB 1: Incident Predictor UI Skeleton
    with tab1:
        st.markdown("""
        <style>
            /* Tab container */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                justify-content: center;
            }
            /* Tab headers */
            .stTabs [data-baseweb="tab"] {
                font-size: 9rem; #text size 
                font-weight: 900;
                padding: 1.5rem 3rem;
                background-color: #b1d1fa;
                color: black;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            /* Hover effect */
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #54b096;
                color: black
                transform: translateY(-2px);
            }
            /* Selected tab */
            .stTabs [aria-selected="true"] {
                font-size: 1.6rem;
                font-weight: 700;
                background-color: #b1d1fa;
                color: black;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)
        # Use the styled title 
        st.markdown("""
            <style>
            .sub-header {
                font-size: 0.5rem;
                font-weight: 100;
                color: #fafbfc;
                margin-top: 0.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
    
        st.markdown('<h2 class="sub-header">Please choose from below options for a prediction! </h2>', unsafe_allow_html=True)
        # Custom CSS for form elements
        st.markdown("""
            <style>
            /* Style for form labels */
            .stForm label {
                font-size: 3rem;
                font-weight: bold;
                color: #080808;
                padding: 0.5rem 0;
            }
        
            /* Style for selectbox labels */
            .stSelectbox label {
                font-size: 1.3rem;
                font-weight: bold;
                color: #080808;
                margin-bottom: 0.5rem;
            }
        
            /* Style for form container */
            .stForm {
                padding: 1rem;
                background-color: #b1d1fa;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)

        flight_details = {
            "origin": None,
            "destination": None,
            "departure_time": None,
            }
        st.markdown("""
        <style>
        .stButton button {
            width: 200px;
            margin: 1rem auto;
            background-color: #4CAF50;
            background: linear-gradient(45deg, #1E88E5, #00BCD4);
            color: white;
            font-size: 2.1rem;
            font-weight: bold;
            padding: 3rem 6rem;
            border-radius: 5px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background: linear-gradient(45deg, #1565C0, #0097A7);
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            </style>
            """, unsafe_allow_html=True)
        with st.form(key="user_flight_details"):
        # create three columns for parallel display
            col1, col2, col3 = st.columns(3)
            with col1: 
                flight_details["origin"] = st.selectbox(label="Enter the departure airport: ", options=unique_origin, placeholder='LGA')
            with col2: 
                flight_details["destination"] = st.selectbox(label="Enter the destination airport: ", options=unique_destination, placeholder='ORF')
            with col3: 
                flight_details["departure_time"] = st.time_input("Enter your departure time (use military time): ")
            # submit button
            submit = st.form_submit_button("Submit")
    
        # Process the form submission
        # Check if the form was submitted
        if submit:
            if not all(flight_details.values()):
                    st.warning("Please fill in all of the fields")
            else:
                # Convert departure_time (which is a time object) to string
                df = pd.DataFrame({
                    "origin": [flight_details["origin"]],
                    "destination": [flight_details["destination"]],
                    "departure_time": [flight_details["departure_time"].strftime("%H:%M")]
                })
                # Check if the route is valid
                df=check_route(df, route_frequency)

            if df is None:
                st.error(f"Invalid Flight Route. Please check your origin and destination.")

            else:
        
                df['Time'] = df['departure_time'].apply(hhmm_to_minutes)
                df['time_sin'] = np.sin(2 * np.pi * df['Time'] / 1440)  # 1440 minutes in a day
                df['time_cos'] = np.cos(2 * np.pi * df['Time'] / 1440)
        
                df = df[['time_sin', 'time_cos', 'route_encoded']]
        
                # make the predictions
                probability = model.predict_proba(df)
                percent_probability = probability[:, 1] * 100
                print(percent_probability)

                # Display predictions
                st.write(f"The probability of your plane crashing is {percent_probability.item():.2f}%")
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=percent_probability.item(),
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': percent_probability.item()
                        }
                    },
                    title={'text': "Flight Incident Risk (%)"}
                ))

                st.plotly_chart(fig, use_container_width=True)
    # TAB 2: Model Performance UI Skeleton
    with tab2:
        st.header("Model Performance Visualizations")
        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display GIFs in respective columns
        with col1:
            st.subheader("ROC curve")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "ROC.png"), use_container_width=True)
    
        with col2:
            st.subheader("Feature Importance")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "feature_importance.png"), use_container_width=True)
        #st.divider()
        st.header("Probability Plots")
        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Calibrated Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "prob_plot_calib.png"), use_container_width=True)
    
        with col2:
            st.subheader("Optimized Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "prob_plot_optimized.png"), use_container_width=True)
        
        st.header("Calibration Plots")
        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Base Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "calib_plot_base.png"), use_container_width=True)
    
        with col2:
            st.subheader("Optimized Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "calib_plot_calib.png"), use_container_width=True)
        
        st.header("Confusion Matrix")
        # Create two columns for parallel display
        col1, col2, col3 = st.columns(3)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Base Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion.png"), use_container_width=True)
    
        with col2:
            st.subheader("Calibrated Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion_calibrated.png"), use_container_width=True)
        
        with col3:
            st.subheader("Optimized Model")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "test_confusion_optimized.png"), use_container_width=True) 
            
    # TAB 3: Data oration UI Skeleton
    with tab3:
        st.header("Data Visualizations")   
        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Incident Feature")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "incidents.png"), use_container_width=True)
    
        with col2:
            st.subheader("Feature Importance")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "cross-correlation-matrix.png"), use_container_width=True)
        #st.divider()
        st.header("Airport Features")
        
        col1, col2, col3 = st.columns(3)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Origin Airports")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "origin_airports.png"), use_container_width=True)
    
        with col2:
            st.subheader("Destination Airports")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "destination_airports.png"), use_container_width=True)
        
        with col3:
            st.subheader("Incident Routes")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "incident_routes.png"), use_container_width=True)
        
        st.header("Time Features")   
        # Create two columns for parallel display
        col1, col2 = st.columns(2)

        # Display GIFs in respective columns
        with col1:
            st.subheader("Cyclical encoding Visualization")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "cyclical_encoding.png"), use_container_width=True)
    
        with col2:
            st.subheader("Time representation in a circle plot")  # Adding a subheader
            st.image(os.path.join(os.path.dirname(__file__), "static", "circle_time.png"), use_container_width=True)
    with tab4:
        st.header("Choose flight for map and animation")
        # Load airports data
        airports = load_airports()
        origin_label, dest_label = tab4func(airports)

    origin = airports[airports['Label'] == origin_label].iloc[0]
    destination = airports[airports['Label'] == dest_label].iloc[0]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Origin", f"{origin['City']} ({origin['IATA']})")
    col2.metric("Destination", f"{destination['City']} ({destination['IATA']})")
    distance = np.round(np.linalg.norm(np.array([origin['Latitude'], origin['Longitude']]) -
                                       np.array([destination['Latitude'], destination['Longitude']])), 2)
    col3.metric("Lat/Lon Distance", f"{distance}°")

    # Shared map data
    curved_path = generate_arc(
        (origin['Longitude'], origin['Latitude']),
        (destination['Longitude'], destination['Latitude']),
        num_points=100,
        height=8
    )
    icon_layer_data = pd.DataFrame([
        {
            "name": f"{origin['City']} ({origin['IATA']})",
            "coordinates": [origin['Longitude'], origin['Latitude']],
            "icon_data": icon_data
        },
        {
            "name": f"{destination['City']} ({destination['IATA']})",
            "coordinates": [destination['Longitude'], destination['Latitude']],
            "icon_data": icon_data
        }
    ])
    view_state = pdk.ViewState(
        latitude=(origin['Latitude'] + destination['Latitude']) / 2,
        longitude=(origin['Longitude'] + destination['Longitude']) / 2,
        zoom=3,
        pitch=0,
    )
    tooltip = {
        "html": "<b>Airport:</b> {name}",
        "style": {"backgroundColor": "black", "color": "white"}
    }
    icon_layer = pdk.Layer(
        type='IconLayer',
        data=icon_layer_data,
        get_icon='icon_data',
        get_size=4,
        size_scale=15,
        get_position='coordinates',
        pickable=True,
    )

    with tab5:
        st.header("Map View")  
        
        curved_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": curved_path}],
            get_path="path",
            get_color=[255, 100, 100],
            width_scale=20,
            width_min_pixels=3,
            get_width=5,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[icon_layer, curved_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip=tooltip
        ))
    with tab6:
        st.header("Flight Animation") 
        chart_placeholder = st.empty()
        start = st.button("🛫 Start Flight")
        if start:
            trail_length = 5
            for i in range(len(curved_path)):
                plane_position = curved_path[i]
                trail = curved_path[max(0, i - trail_length):i + 1]
                plane_layer = pdk.Layer(
                    "TextLayer",
                    data=[{"position": plane_position, "text": "✈️"}],
                    get_position="position",
                    get_text="text",
                    get_size=32,
                    get_angle=0,
                    get_color=[0, 0, 0],
                )
                trail_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": trail}],
                    get_path="path",
                    get_color=[50, 50, 255],
                    width_scale=20,
                    width_min_pixels=2,
                    get_width=3,
                )
                curved_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": curved_path}],
                    get_path="path",
                    get_color=[255, 100, 100],
                    width_scale=20,
                    width_min_pixels=3,
                    get_width=5,
                )
                r = pdk.Deck(
                    layers=[icon_layer, curved_layer, trail_layer, plane_layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v9",
                    tooltip=tooltip
                )
                chart_placeholder.pydeck_chart(r)
                time.sleep(0.05)
if __name__ == "__main__":
    main()