import streamlit as st
import pickle
import datetime
import pandas as pd
import numpy as np
import functions
import plotly.express as px





# App configs
st.set_page_config(
page_title="Bike Sharing Demand Prediction",
layout="wide",
initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
        margin-up: 200px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: 200px;
        margin-up: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# sidebar, navigation
st.sidebar.write("<h1><b> Navigation</b></h1>", unsafe_allow_html=True)
nav = st.sidebar.radio("", ["Home", "Clustering"])
st.sidebar.write("<br>", unsafe_allow_html=True)

if nav == 'Home':


    #st.write(functions.get_treat_48h_data()) # api request for next 48h

    # Heading
    # st.markdown("<h1 style='text-align: center; background-color:darkgrey'>🚴 Bike Rental Demand Prediction 🚴</h1>", 
    #             unsafe_allow_html=True)
    # # Sub heading
    # st.markdown("<h4 style='text-align: center'><i>∞∞∞ A Machine Learning based web app to predict bike rental demand ∞∞∞</i></h4>",
    #             unsafe_allow_html=True)
    # # Image
    # st.markdown("<h1 align='center'><img src='https://storage.googleapis.com/kaggle-competitions/kaggle/3948/media/bikes.png'></img></h1>", 
    #             unsafe_allow_html=True)



    # About 
    # st.write("Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis.")
    # st.write("This project is based on a Kaggle competition. Our task is to combine historical usage patterns with weather data in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C.")
    # st.markdown("<i>For more details on this competition, [visit here](https://www.kaggle.com/c/bike-sharing-demand).</i>", unsafe_allow_html=True)

    st.markdown("<br><h4><b> Please fill in the below details:</b></h4><br>", unsafe_allow_html=True)

    st.sidebar.write("<h1><b>Predictions</b></h1>", unsafe_allow_html=True)
    # User input features
    c1, c2, c3 = st.columns([1.5,3,3])
    with c1:
        with st.expander("custom predict"):
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
            button_custom_predict = st.button("Predict Rentals")
            success1 = st.container()
    
    with c2:
        with st.expander("prédiction sur les prochaines 24h"):
            

            button_day_predict = st.button("prédire sur la fin de la journée")
            
            button_48h_predict = st.button("prédire sur les prochaines 48h")
            success2 = st.container()
    with c3:
        with st.expander("prédiction sur une date ultérieure"):
            date2 = st.date_input("Enter date:")
            button_date_predict = st.button("prédire 24h par date")
            success3 = st.container()

    if button_custom_predict:
        if ((date=='') | (time=='') | (weather=='') | 
            (temp=='') | (humidity=='') | (windspeed=='')):
            success1.error("Please fill all fields before proceeding.")
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
            success1.success("La prédiction est de " + str(prediction) + " vélos")

    if button_day_predict:
        if ((date=='')):
            st.error("Please fill the date field before proceeding.")
        else :
            # You will have to create the model
            savedmodel = open('pickled_model.sav', 'rb')
            model = pickle.load(savedmodel)
            savedmodel.close()

            df = functions.get_treat_48h_data()
            prediction = np.round(np.abs(np.expm1(model.predict(df))))
            prediction = pd.DataFrame(prediction)
            print(df.info())
            pred = prediction.join(df[["hour","day"]])
            pred = pred.loc[pred['day'] == dt.day]
            pred = pred.drop('day',axis = 1)
            pred = pred.rename(columns={0: "Prédiction"})

            titre = 'Prévision du nombre de Vlille heure par heure pour le ' + str(dt.strftime('%x'))
            fig = fig = px.line(pred, x = "hour", y = "Prédiction", title = titre)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'height' : 596,
               'width' : 650
                })
            success2.plotly_chart(fig,width=1200,height=1200)


    if button_48h_predict:

        # You will have to create the model
        savedmodel = open('pickled_model.sav', 'rb')
        model = pickle.load(savedmodel)
        savedmodel.close()

        df = functions.get_treat_48h_data()
        prediction = np.round(np.abs(np.expm1(model.predict(df))))
        prediction = pd.DataFrame(prediction)
        #st.write(df)
        pred = prediction.join(df[["hour","day"]])
        pred = pred.rename(columns={0: "Prédiction"})

        titre = 'Prévision du nombre de Vlille heure par heure pour les 48 prochaines heures'
        fig = fig = px.line(pred, x = "hour", y = "Prédiction", title = titre, color='day')
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'height' : 596,
            'width' : 650
            })
        success2.plotly_chart(fig,width=1200,height=1200)

    if button_date_predict:
        if ((date2=='')):
            st.error("Please fill the date field before proceeding.")
        else :
            # You will have to create the model
            savedmodel = open('pickled_model.sav', 'rb')
            model = pickle.load(savedmodel)
            savedmodel.close()

            df = functions.get_treat_7days_data(str(date2))
            prediction = np.round(np.abs(np.expm1(model.predict(df))))
            prediction = pd.DataFrame(prediction)
            
            pred = prediction.join(df[["hour","day"]])
            pred = pred.loc[pred['day'] == date2.day]
            pred = pred.drop('day',axis = 1)
            pred = pred.rename(columns={0: "Prédiction"})
            
            titre = 'Prévision du nombre de Vlille heure par heure pour le ' + str(dt.strftime('%x'))
            fig = fig = px.line(pred, x = "hour", y = "Prédiction", title = titre)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'height' : 596,
               'width' : 650
                })
            success3.plotly_chart(fig,width=1200,height=1200)
if nav == 'Clustering':
    st.write("Here you are")
    functions.clustering()
    

















