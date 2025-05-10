# functions/tab_one.py

import streamlit as st
import pandas as pd

def tab1(airports: pd.DataFrame, model, encoder, frequencies: dict):
    '''Defines incident prediction tab.'''
    st.header("Predict Flight Incident Risk")
    col1, col2 = st.columns(2)
    with col1:
        origin_label = st.selectbox("Origin Airport", airports['Label'])
        dest_label = st.selectbox("Destination Airport", airports['Label'])
    with col2:
        departure_time = st.number_input("Departure Time (e.g., 1430)", min_value=0, max_value=2359, step=5)

    origin_iata = airports[airports['Label'] == origin_label]['IATA'].values[0]
    dest_iata = airports[airports['Label'] == dest_label]['IATA'].values[0]

    if st.button("Predict Incident Probability"):
        try:
            input_df = pd.DataFrame({
                "Origin": [origin_iata],
                "Destination": [dest_iata],
                "DepTime": [departure_time]
            })
            encoded_input = encoder.transform(input_df)
            prob = model.predict_proba(encoded_input)[0][1]
            st.success(f"Estimated Incident Probability: {prob:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    return origin_label, dest_label, departure_time