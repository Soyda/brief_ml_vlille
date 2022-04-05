import requests
import pandas as pd
import numpy as np
import datetime as dt

def get_treat_48h_data():

    forecast = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=50.62925&lon=3.057256&exclude=minutely&units=metric&appid=39774e56fd9f1b795ec7ced21c030cd1").json()
    forecast_df = pd.json_normalize(forecast["hourly"])

    datetime_list = []
    for i in forecast_df["dt"] :
        datetime_list.append(dt.datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))
    forecast_df["dt"] = datetime_list

    forecast_df = forecast_df.fillna(0)

    weather_cat = {1:["01d", "01n", "02d", "02n", "03d", "03n"], 2:["04d", "04n", "50d", "50n"], 3:["09d", "09n", "10d", "10n"], 4:["11d", "11n", "13d", "13n"]}
    weather_cat_list = []
    for i in forecast_df["weather"]:
        for j in weather_cat.keys():
            if i[0]["icon"] in weather_cat[j]:
                weather_cat_list.append(j)
    forecast_df["weather"] = weather_cat_list

    forecast_df['dt'] = pd.to_datetime(forecast_df['dt'])
    forecast_df["month"] = forecast_df["dt"].dt.month
    forecast_df["year"] = forecast_df["dt"].dt.year
    forecast_df["hour"] = forecast_df["dt"].dt.hour
    forecast_df["week_day"] = forecast_df["dt"].dt.strftime("%A")

    forecast_df['MM-DD'] = pd.to_datetime(forecast_df['dt']).dt.strftime('%m-%d')

    forecast_df.loc[forecast_df['MM-DD'] < '12-31','saison'] = 1
    forecast_df.loc[forecast_df['MM-DD'] < '12-21','saison'] = 4
    forecast_df.loc[forecast_df['MM-DD'] < '09-21','saison'] = 3
    forecast_df.loc[forecast_df['MM-DD'] < '06-21','saison'] = 2
    forecast_df.loc[forecast_df['MM-DD'] < '03-31','saison'] = 1

    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Monday", 0, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Tuesday", 1, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Wednesday", 2, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Thursday", 3, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Friday", 4, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Saturday", 5, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Sunday", 6, forecast_df['week_day'])

    holidays = ["01-01",
                "05-01",
                "05-08",
                "07-14",
                "08-15",
                "11-01",
                "11-11",
                "12-25"]

    is_holiday = []
    for i in forecast_df["MM-DD"]:
        if i in holidays :
            is_holiday.append(1)
        else :
            is_holiday.append(0)

    forecast_df["holiday"] = is_holiday

    forecast_df = forecast_df.rename(columns={"feels_like": "atemp", "wind_speed": "windspeed"})

    forecast_df = forecast_df[["holiday", "weather", "temp", "atemp", "humidity", "windspeed", "month", "year", "hour", "saison", "week_day"]]

    return forecast_df