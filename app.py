import streamlit as st
import pickle
import datetime
import pandas as pd
import numpy as np
import functions

def temp_ressentie(temp,hum):
    temp_r = (-8.785) + 1.611*temp + 2.339*hum + 	(-0.146*temp*hum) + (-1.231*10**-2)*temp**2 + (-1.642*10**-2)*hum**2 + (2.212*10**-3)*(temp**2)*hum + (7.255*10**-4)*temp*(hum**2) + (-3.582*10**-6)*(temp**2)*(hum**2)
    
    return str(temp_r)

def buid_df_graph(df_meteo,df):
    return 0



# App configs
st.set_page_config(
page_title="Bike Sharing Demand Prediction",
layout="centered",
initial_sidebar_state="expanded",
)

st.write(functions.get_treat_48h_data()) # api request for next 48h

# Heading
st.markdown("<h1 style='text-align: center; background-color:deepskyblue'>🚴 Bike Rental Demand Predoction 🚴</h1>", 
            unsafe_allow_html=True)
# Sub heading
st.markdown("<h4 style='text-align: center'><i>∞∞∞ A Machine Learning based web app to predict bike rental demand ∞∞∞</i></h4>",
            unsafe_allow_html=True)
# Image
st.markdown("<h1 align='center'><img src='https://storage.googleapis.com/kaggle-competitions/kaggle/3948/media/bikes.png'></img></h1>", 
            unsafe_allow_html=True)



# About 
st.write("Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis.")
st.write("This project is based on a Kaggle competition. Our task is to combine historical usage patterns with weather data in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C.")
st.markdown("<i>For more details on this competition, [visit here](https://www.kaggle.com/c/bike-sharing-demand).</i>", unsafe_allow_html=True)

st.markdown("<br><h4><b> Please fill in the below details:</b></h4><br>", unsafe_allow_html=True)

# User input features
date = st.date_input("Enter date :")
time = st.time_input("Enter Time (HH24:MM):")
dt = datetime.datetime.combine(date,time)
#day = st.selectbox("What type of day is it?", ['Holiday', 'Working day', 'Weekend'])
holiday = st.checkbox("Is it a special holiday ?")

display = ("Clear/Few clouds", "Mist/Cloudy","Light Rain/Light Snow/Scattered clouds","Heavy Rain/Snowfall/Foggy/Thunderstorm")
options = list(range(len(display)))
weather = st.selectbox("What type of weather is it?", options, format_func=lambda x: display[x]) +1

temp = st.text_input("Enter temperature (in °C):")
humidity = st.text_input("Enter humidity (in %):")
windspeed = st.text_input("Enter windspeed (in km/h):")

if st.button("Predict Rentals"):
    if ((date=='') | (time=='') | (weather=='') | 
        (temp=='') | (humidity=='') | (windspeed=='')):
        st.error("Please fill all fields before proceeding.")
    else :
        # You will have to create the model
        savedmodel = open('pickled_model.sav', 'rb')
        model = pickle.load(savedmodel)
        savedmodel.close()

        hour = dt.hour
        week_day = dt.weekday()
        month = dt.month
        year = dt.year
        workingday = 0 if holiday or week_day >4 else 1
        holiday = 1 if holiday else 0 
        atemp = float(temp) + 3 if float(windspeed) < 12 else float(temp) -1
        md = dt.strftime('%m-%d')
        if md < '12-31':
            saison = 1.0
        if md < '12-21':
            saison = 4.0
        if md < '09-21':
            saison = 3.0
        if md < '06-21':
            saison = 2.0
        if md < '03-21':
            saison = 1.0

        df= {

            "holiday" : [holiday],
            #'workingday':[workingday],
            'weather':[weather],
            'temp':[temp],
            'atemp':[atemp],
            'humidity':[humidity],
            'windspeed':[windspeed],
            'month':[month],
            'year':[year],
            'hour':[hour],
            'saison':[saison],
            'week_day':[week_day],

            }
        df = pd.DataFrame.from_dict(df)
        prediction = int(np.abs(np.expm1(model.predict(df))))
        #st.success("There will be an approx. demand of " + str(prediction) + " bikes for above conditions.")
        st.success("La prédiction est de " + str(prediction) + " vélos")


if st.button("Predict entire day"):
    if ((date=='')):
        st.error("Please fill the date field before proceeding.")
    else :
        # You will have to create the model
        savedmodel = open('pickled_model.sav', 'rb')
        model = pickle.load(savedmodel)
        savedmodel.close()

        df = functions.get_treat_48h_data(dt.date)

        df = pd.DataFrame.from_dict(df)
        prediction = int(np.abs(np.expm1(model.predict(df))))
        #st.success("There will be an approx. demand of " + str(prediction) + " bikes for above conditions.")
        st.success("La prédiction est de " + str(prediction) + " vélos")
        st.line_chart(data=range(0,20,1), width=0, height=0, use_container_width=True)





















