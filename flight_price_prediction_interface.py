import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Flight Price Prediction", layout="wide")

st.sidebar.header("About ✈️")
st.sidebar.info("\n**Welcome to Flight Price Prediction!**\n\n🚀 Enter your flight details below and click **Predict** to get an estimated ticket price.\n\n💡 This tool uses machine learning to provide accurate price predictions based on historical flight data.")

# Flight Details Form
with st.form(key='flight_form'):
    st.subheader("Flight Date & Time ⏳")
    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Select Flight Date:")
    with col2:
        departure_time = st.time_input("Departure Time:")
    with col3:
        arrival_time = st.time_input("Arrival Time:")
    
    st.subheader("Source & Destination 🌍")
    col_source, col_dest = st.columns(2)
    with col_source:
        source = st.selectbox("Choose Source:", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'])
    with col_dest:
        destination = st.selectbox("Choose Destination:", ['New Delhi', 'Bangalore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    
    # Flight Timing Validation
    min_travel_time = {
        ('Mumbai', 'Kolkata'): 2.5,
        ('Delhi', 'Bangalore'): 2.0,
        ('Chennai', 'Hyderabad'): 1.5,
        ('Kolkata', 'Delhi'): 2.5,
        ('Mumbai', 'Delhi'): 2.0,
        ('Bangalore', 'Kolkata'): 3.0,
        ('Chennai', 'Delhi'): 2.5,
        ('Hyderabad', 'Kolkata'): 2.0,
        ('Bangalore', 'Bangalore'): 0.0,
        ('Bangalore', 'New Delhi'): 2.5,
        ('Kolkata', 'Kolkata'): 0.0,
        ('Kolkata', 'New Delhi'): 2.5,
        ('Delhi', 'New Delhi'): 0.5,
        ('Delhi', 'Delhi'): 0.0,
        ('Chennai', 'New Delhi'): 3.0,
        ('Chennai', 'Bangalore'): 1.0,
        ('Chennai', 'Cochin'): 1.5,
        ('Chennai', 'Kolkata'): 2.5,
        ('Chennai', 'Hyderabad'): 1.5,
        ('Mumbai', 'New Delhi'): 2.5,
        ('Mumbai', 'Bangalore'): 2.0,
        ('Mumbai', 'Cochin'): 1.5,
        ('Mumbai', 'Kolkata'): 2.5,
        ('Mumbai', 'Delhi'): 2.0,
        ('Mumbai', 'Hyderabad'): 1.5
    }
    duration = (pd.to_datetime(str(arrival_time)) - pd.to_datetime(str(departure_time))).seconds / 3600
    invalid_timing = False
    error_message = ""
    if (source, destination) in min_travel_time and duration < min_travel_time[(source, destination)]:
        error_message = f"Invalid timing: Minimum travel time from {source} to {destination} is {min_travel_time[(source, destination)]} hours."
        invalid_timing = True
    
    st.subheader("Flight Stops & Airline ✈️")
    total_stops = st.selectbox("Select Total Stops:", list(range(5)))
    airline = st.selectbox("Choose Airline:", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia', 'Vistara Premium economy', 'Jet Airways Business', 'Multiple carriers Premium economy', 'Trujet'])
    
    st.subheader("Additional Flight Information ℹ️")
    add_info_options = ['No info 1', 'No info 2', 'In-flight meal not included', 'No check-in baggage included', '1 Short layover', '1 Long layover', 'Change airports', 'Business class', 'Red-eye flight', '2 Long layover']
    add_info = st.multiselect("Select Additional Info:", add_info_options)
    
    # Ensure "No info" logic is handled correctly
    if 'No info 1' in add_info or 'No info 2' in add_info:
        add_info = ['No info 1', 'No info 2']  # Restrict to only No info options
    
    # Submit Button
    submitted = st.form_submit_button("Predict ✨")

if submitted and not invalid_timing:
    @st.cache_resource
    def load_model():
        return joblib.load('flight_price_predictor.pkl')
    model = load_model()

    # Feature encoding
    input_data = [
        total_stops, date.day, date.month, date.year, departure_time.hour, departure_time.minute,
        arrival_time.hour, arrival_time.minute, duration, 0,
        *(1 if airline == a else 0 for a in ['Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']),
        *(1 if source == s else 0 for s in ['Bangalore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']),
        *(1 if destination == d else 0 for d in ['New Delhi', 'Bangalore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']),
        *(1 if i in add_info else 0 for i in ['1 Long layover', '1 Short layover', '2 Long layover', 'Business class', 'Change airports', 'In-flight meal not included', 'No check-in baggage included', 'No info 1', 'No info 2', 'Red-eye flight'])
    ]

    price = model.predict([input_data])
    st.success(f"Prediction complete! 🏆")
    st.markdown(f"<h3 style='color: green;'>Predicted Flight Price: ₹{price[0]:,.2f}</h3>", unsafe_allow_html=True)

if submitted and invalid_timing:
    st.error(error_message)

if st.sidebar.button("View Travel Time Estimates 🕒"):
    st.title("Estimated Travel Time for Flight Routes ⏳")
    df_travel = pd.DataFrame([(f"{src} ➝ {dest}", f"{time} hours") for (src, dest), time in min_travel_time.items()], columns=["Route", "Approximate Time"])
    st.dataframe(df_travel.style.set_properties(**{'background-color': 'black', 'color': 'white', 'border-color': 'red'}))
