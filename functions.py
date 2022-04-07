import requests
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

def get_treat_48h_data():
    """Function that request on an api to get weather for the next 48h in Lille.
        It returns formated dataframe.
        It doesn't take any argument."""

    # API request
    forecast = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=50.62925&lon=3.057256&exclude=minutely&units=metric&appid=39774e56fd9f1b795ec7ced21c030cd1").json()
    forecast_df = pd.json_normalize(forecast["hourly"])

    # convert dates to a more readable format for human
    datetime_list = []
    for i in forecast_df["dt"] :
        datetime_list.append(dt.datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))
    forecast_df["dt"] = datetime_list

    # Replace NaN to zeros
    forecast_df = forecast_df.fillna(0)

    # create weather categories
    weather_cat = {1:["01d", "01n", "02d", "02n", "03d", "03n"], 2:["04d", "04n", "50d", "50n"], 3:["09d", "09n", "10d", "10n"], 4:["11d", "11n", "13d", "13n"]}
    weather_cat_list = []
    for i in forecast_df["weather"]:
        for j in weather_cat.keys():
            if i[0]["icon"] in weather_cat[j]:
                weather_cat_list.append(j)
    forecast_df["weather"] = weather_cat_list

    # create columns month, year, hour, and week_day
    forecast_df['dt'] = pd.to_datetime(forecast_df['dt'])
    forecast_df["month"] = forecast_df["dt"].dt.month
    forecast_df["year"] = forecast_df["dt"].dt.year
    forecast_df["hour"] = forecast_df["dt"].dt.hour
    forecast_df["week_day"] = forecast_df["dt"].dt.strftime("%A")

    # create de month-day column to affect a number corresponding to the season
    forecast_df['MM-DD'] = pd.to_datetime(forecast_df['dt']).dt.strftime('%m-%d')

    # affecting seasons
    forecast_df.loc[forecast_df['MM-DD'] < '12-31','saison'] = 1
    forecast_df.loc[forecast_df['MM-DD'] < '12-21','saison'] = 4
    forecast_df.loc[forecast_df['MM-DD'] < '09-21','saison'] = 3
    forecast_df.loc[forecast_df['MM-DD'] < '06-21','saison'] = 2
    forecast_df.loc[forecast_df['MM-DD'] < '03-31','saison'] = 1

    # replacing the day by a number
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Monday", 0, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Tuesday", 1, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Wednesday", 2, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Thursday", 3, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Friday", 4, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Saturday", 5, forecast_df['week_day'])
    forecast_df['week_day'] = np.where(forecast_df['week_day'] == "Sunday", 6, forecast_df['week_day'])

    # list of the fixed holidays dates
    holidays = ["01-01",
                "05-01",
                "05-08",
                "07-14",
                "08-15",
                "11-01",
                "11-11",
                "12-25"]

    # checking holidays dates in data
    is_holiday = []
    for i in forecast_df["MM-DD"]:
        if i in holidays :
            is_holiday.append(1)
        else :
            is_holiday.append(0)

    # creating holiday column (1 for True)
    forecast_df["holiday"] = is_holiday

    # rename columns to correspond to our datas
    forecast_df = forecast_df.rename(columns={"feels_like": "atemp", "wind_speed": "windspeed"})

    # only keeping the columns we want to use
    forecast_df = forecast_df[["holiday", "weather", "temp", "atemp", "humidity", "windspeed", "month", "year", "hour", "saison", "week_day"]]

    return forecast_df


def get_treat_7days_data(arg_date):

    arg_date = dt.datetime.strptime(arg_date, '%Y-%m-%d')

    forecast_day = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat=50.62925&lon=3.057256&exclude=minutely&units=metric&appid=39774e56fd9f1b795ec7ced21c030cd1").json()
    forecast_day_df = pd.json_normalize(forecast_day["daily"])

    datetime_list = []
    for i in forecast_day_df["dt"] :
        datetime_list.append(dt.datetime.fromtimestamp(i).strftime('%Y-%m-%d'))
    forecast_day_df["dt"] = datetime_list
    
    weather_cat = {1:["01d", "01n", "02d", "02n", "03d", "03n"], 2:["04d", "04n", "50d", "50n"], 3:["09d", "09n", "10d", "10n"], 4:["11d", "11n", "13d", "13n"]}
    weather_cat_list = []
    for i in forecast_day_df["weather"]:
        for j in weather_cat.keys():
            if i[0]["icon"] in weather_cat[j]:
                weather_cat_list.append(j)
    forecast_day_df["weather"] = weather_cat_list

    forecast_day_df["dt"] = pd.to_datetime(forecast_day_df["dt"])

    date_df = forecast_day_df[forecast_day_df["dt"] == arg_date ]

    date_df["hour"] = 0
    
    for j in range(23):
        date_df = date_df.append(date_df.iloc[0], ignore_index=True)
    
    date_df["hour"] = [j for j in range(24)]

    temp_list = []
    atemp_list = []
    for h in date_df["hour"] :
        if h < 6 or h >= 22 :
            temp_list.append(date_df[date_df["hour"] == h]["temp.night"][h])
            atemp_list.append(date_df[date_df["hour"] == h]["feels_like.night"][h])
        elif h > 6 and h <= 12 :
            temp_list.append(date_df[date_df["hour"] == h]["temp.morn"][h])
            atemp_list.append(date_df[date_df["hour"] == h]["feels_like.morn"][h])
        elif h > 12 and h < 18 :
            temp_list.append(date_df[date_df["hour"] == h]["temp.day"][h])
            atemp_list.append(date_df[date_df["hour"] == h]["feels_like.day"][h])
        else :
            temp_list.append(date_df[date_df["hour"] == h]["temp.eve"][h])
            atemp_list.append(date_df[date_df["hour"] == h]["feels_like.eve"][h])
    date_df["temp"] = temp_list
    date_df["atemp"] = atemp_list

    date_df = date_df[["dt", "humidity", "wind_speed", "weather", "temp", "atemp", "hour"]]

    date_df["month"] = date_df["dt"].dt.month
    date_df["year"] = date_df["dt"].dt.year
    date_df["week_day"] = date_df["dt"].dt.strftime("%A")

    date_df['MM-DD'] = pd.to_datetime(date_df['dt']).dt.strftime('%m-%d')

    date_df.loc[date_df['MM-DD'] < '12-31','saison'] = 1
    date_df.loc[date_df['MM-DD'] < '12-21','saison'] = 4
    date_df.loc[date_df['MM-DD'] < '09-21','saison'] = 3
    date_df.loc[date_df['MM-DD'] < '06-21','saison'] = 2
    date_df.loc[date_df['MM-DD'] < '03-31','saison'] = 1

    date_df['week_day'] = np.where(date_df['week_day'] == "Monday", 0, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Tuesday", 1, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Wednesday", 2, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Thursday", 3, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Friday", 4, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Saturday", 5, date_df['week_day'])
    date_df['week_day'] = np.where(date_df['week_day'] == "Sunday", 6, date_df['week_day'])

    holidays = ["01-01",
                "05-01",
                "05-08",
                "07-14",
                "08-15",
                "11-01",
                "11-11",
                "12-25"]

    is_holiday = []
    for i in date_df["MM-DD"]:
        if i in holidays :
            is_holiday.append(1)
        else :
            is_holiday.append(0)

    date_df["holiday"] = is_holiday

    date_df = date_df.rename(columns={"wind_speed": "windspeed"})

    date_df = date_df[["holiday", "weather", "temp", "atemp", "humidity", "windspeed", "month", "year", "hour", "saison", "week_day"]]

    return date_df


# @st.experimental_memo(suppress_st_warning=True)
def clustering_kmeans():
    """Function that do a clustering using k means, on our data
    """
    st.title("Clustering with K Means")

    st.write("Here we want to do a clustering using the weather parameters such as temperature, apparent temperature, humidity, and windspeed.")
    st.write("Our data having a weather column divided in 4 categories, our clustering is meant to see if we obtain the same clusters as in the weather column.")

    st.write("We load the data :")
    # loading data
    df = pd.read_csv("src/dataset_cleaned.csv")
    st.dataframe(df)

    # keeping data about weather
    df_kmeans = df[["temp", "humidity", "windspeed", "atemp"]]
    st.write("We're going to work with this part of the data :")
    st.dataframe(df_kmeans)

    # PCA
    st.write("Let's first do a PCA, this will decrease the number of dimension and allow us to see which components explain the variance the most :")
    X = df_kmeans.values
    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(X)
    # Create a PCA instance: pca
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(X_std)

    # Plot the explained variances
    features = range(pca.n_components_)

    # c1, c2, c3 = st.columns([1,3,1])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar(features, pca.explained_variance_ratio_, color='black')
    ax.set_xlabel("PCA features")
    ax.set_ylabel("variance %")
    ax.set_xticks(features)
    st.write(fig)
    st.write("By reading the graph we can see that the first two components explain about 80'%' of the variance.")

    # Save components to a DataFrame
    PCA_components = pd.DataFrame(principalComponents)

    st.write("Let's plot some visualization of the PCA components in function of each other :")
    # plot PCA1 in function of PCA0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    ax.set_xlabel("PCA 0")
    ax.set_ylabel("PCA 1")
    st.write(fig)

    with st.expander("See more"):
    # plot PCA2 in function of PCA0
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(PCA_components[0], PCA_components[2], alpha=.1, color='black')
        ax.set_xlabel("PCA 0")
        ax.set_ylabel("PCA 2")
        st.write(fig)

        # plot PCA1 in function of PCA2
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(PCA_components[1], PCA_components[2], alpha=.1, color='black')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.write(fig)

    # K Means
    st.write("Now let's use K-Means method to find our clusters.")
    st.write("First let's plot an elbow curve to estimate the optimal number of clusters :")
    # elbow curve
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(PCA_components.iloc[:,:3])
        
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
    
    # plot elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ks, inertias, '-o', color='black')
    ax.set_xlabel('number of clusters, k')
    ax.set_ylabel('inertia')
    ax.set_xticks(ks)
    st.write(fig)

    st.write("Here we can deduct that the best number of cluster is 4.")

    kmeans = KMeans(n_clusters= 4)
    pred = kmeans.fit_predict(PCA_components)

    # visualizing clusters
    st.write("After applying the 4 clusters we plot again our graphs of the PCA components, but this time we'll see the different clusters")
    # plot PCA1 in function of PCA0
    centers = kmeans.cluster_centers_
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(PCA_components.values[:, 0], PCA_components.values[:, 1], c=pred, s=50, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    ax.set_xlabel("PCA 0")
    ax.set_ylabel("PCA 1")
    st.write(fig)

    with st.expander("See more"):
        # plot PCA2 in function of PCA0
        centers = kmeans.cluster_centers_
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(PCA_components.values[:, 0], PCA_components.values[:, 2], c=pred, s=50, cmap='viridis')
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        ax.set_xlabel("PCA 0")
        ax.set_ylabel("PCA 2")
        st.write(fig)

        # plot PCA1 in function of PCA2
        centers = kmeans.cluster_centers_
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(PCA_components.values[:, 1], PCA_components.values[:, 2], c=pred, s=50, cmap='viridis')
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.write(fig)

    st.write("We add the columns containing the clusters (from 0 to 3) to our dataframe :")
    # concat to dataframe
    df_cluster = pd.concat([df_kmeans, PCA_components.iloc[:,:3]], axis=1)
    df_cluster.columns.values[-3:] = ["PCA_0", "PCA_1", "PCA_2"]
    df_cluster["cluster"] = kmeans.labels_
    st.dataframe(df_cluster)

    # groupby cluster
    # st.write(df_cluster.groupby("cluster").agg({k:"mean" for k in df_cluster.columns}))

    # adding weather column to compare
    df_compare = df[["temp", "humidity", "windspeed", "atemp", "weather"]]
    df_compare["cluster"] = kmeans.labels_

    st.write("Now we can compare our clusters with the original weather category to check for any similitude :")
    # groupby cluster and weather to compare
    st.write("Grouped by cluster :")
    st.dataframe(df_cluster.groupby("cluster").agg({k:"mean" for k in df_cluster.columns}))
    st.write("Number of values by cluster :")
    st.dataframe(df_cluster["cluster"].value_counts())
    st.write("It's noticeable that our clusters are almost homogeously splitted :")
    st.write("  - we have two clusters with rather cool temperatures which differentiate themselves by wind speed and humidity")
    st.write("  - we have two other clusters with rather warm temperatures which also differentiate themselves by wind speed and humidity")

    st.write("Grouped by weather :")
    st.dataframe(df_compare.groupby("weather").agg({k:"mean" for k in df_compare.columns}))
    st.write("- 1: Clear, Few clouds, Partly cloudy, Partly cloudy")
    st.write("- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist")
    st.write("- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds")
    st.write("- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog ")

    st.write("Number of values by weather's category :")
    st.dataframe(df_compare["weather"].value_counts())
    st.write("Here it's noticeable that the weather category is not homogeously splitted :")
    st.write("  - 66% of the rows are considered as category 1, 26% are category 2, 8% are category 3 and only one row is category 4")
    st.write("  - the only paramater which seems to influence the weather category is humidity")
