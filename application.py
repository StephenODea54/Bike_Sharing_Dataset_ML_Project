##### Imports
# UI Interface
import streamlit as st

# Data Manipulation
import pandas as pd
import numpy as np

# To Import Model
import joblib

# To Predict Using the Import Model
from sklearn.ensemble import RandomForestRegressor

### Functions
# Collect User Input and Put into DataFrame
def inputs():
    # Interface for User Input
    season = st.sidebar.selectbox('Season', ('Winter', 'Spring', 'Summer', 'Fall'))
    hr = st.sidebar.selectbox('Hour', ('12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM'))
    weekday = st.sidebar.selectbox('Day of the Week', ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
    workingday = st.sidebar.selectbox('Holiday or Weekend?', ('Yes', 'No'))
    hum = st.sidebar.number_input(label = 'Humidity (%)', min_value = 0, max_value = 1)
    weathersit = st.sidebar.selectbox('Weather', ('Clear, Few Clouds, Partly Cloudy', 'Mist + Cloudy, Mist + Broken Clouds, Mist + Few Clouds, Mist', 'Light Snow, Light Rain + Thunderstorm + Scattered Clouds, Light Rain + Scattered Clouds'))
    temp = st.sidebar.number_input(label = 'Temperature (F)', min_value = 17.6, max_value = 102.2)
    windspeed = st.sidebar.number_input(label = 'Windspeed (MPH)', min_value = 0, max_value = 67)

    ### Transform Inputs for Appropriate Use in Model
    # Temperature: Convert to Celcius and Normalize (t_min = -8 C, t_max = 39 C)
    temp_Celcius = (temp - 32) * (5/9)
    temp_normalized = (temp_Celcius + 8) / (39 + 8)

    # Windspeed: Normalize to 67
    windspeed_normalized = windspeed / 67

    # Map Categorical Inputs
    season_mapper = {'Winter': 1,
                     'Spring': 2,
                     'Summer': 3,
                     'Fall': 4}

    hr_mapper = {'12 AM': 0,
                 '1 AM': 1,
                 '2 AM': 2,
                 '3 AM': 3,
                 '4 AM': 4,
                 '5 AM': 5,
                 '6 AM': 6,
                 '7 AM': 7,
                 '8 AM': 8,
                 '9 AM': 9,
                 '10 AM': 10,
                 '11 AM': 11,
                 '12 PM': 12,
                 '1 PM': 13,
                 '2 PM': 14,
                 '3 PM': 15,
                 '4 PM': 16,
                 '5 PM': 17,
                 '6 PM': 18,
                 '7 PM': 19,
                 '8 PM': 20,
                 '9 PM': 21,
                 '10 PM': 22,
                 '11 PM': 23}

    weekday_mapper = {'Sunday': 0,
                      'Monday': 1,
                      'Tuesday': 2,
                      'Wednesday': 3,
                      'Thursday': 4,
                      'Friday': 5,
                      'Saturday': 6}

    workingday_mapper = {'Yes': 0,
                         'No': 1}

    weathersit_mapper = {'Clear, Few Clouds, Partly Cloudy': 1,
                         'Mist + Cloudy, Mist + Broken Clouds, Mist + Few Clouds, Mist': 2,
                         'Light Snow, Light Rain + Thunderstorm + Scattered Clouds, Light Rain + Scattered Clouds': 3}

    # Put User Inputs into Dict
    data = {'season': season,
            'hr': hr,
            'weekday': weekday,
            'workingday': workingday,
            'hum': hum,
            'weathersit': weathersit,
            'temp': temp_normalized,
            'windspeed': windspeed_normalized}

    # Create DataFrame from Dictionary
    # Use index = [0] since all are scalar values
    attributes = pd.DataFrame(data, columns = ['season', 'hr', 'weekday', 'workingday', 'hum', 'weathersit', 'temp', 'windspeed'], index = [0])

    # Map Levels onto Attributes
    attributes['season'] = attributes['season'].map(season_mapper)
    attributes['hr'] = attributes['hr'].map(hr_mapper)
    attributes['weekday'] = attributes['weekday'].map(weekday_mapper)
    attributes['workingday'] = attributes['workingday'].map(workingday_mapper)
    attributes['weathersit'] = attributes['weathersit'].map(weathersit_mapper)

    # Return DataFrame
    return attributes

### App
# Title
st.title('Bike Prediction App')

# Body
st.write("""

This app predicts the number of bicycle rentals for any given time of day! In order to make a prediction, please provide the relevant information for each of the fields to the left. The algorithm will automatically update everytime an input is changed.

Please note that the minimum and maximum values for temperature are 17.6 and 102.2, resepectively. Similarly, the minimum and maximum values for windspeed are 0 and 67.

The data was made publicly available by the company Capital-Bikeshare in Washington D.C. Since making the data public, Fanaee-T, Hadi, and Joao Gama have added weather and seasonal information to the dataset.
The data may be found at [Bike-Sharing Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

The underlying model used for predictions is a Gradient Boosted Tree. For more information on the model and methodology, the relevant files may be found at my github at [GitHub](https://github.com/StephenODea54/Bike_Sharing_Dataset_ML_Project)

""")

# Sidebar
st.sidebar.header('User Input')

# Extract DataFrame from Function
attributes = inputs()

# Read in GLM
gb = joblib.load('gb_results.pkl')

# Make Predictions
prediction = gb.predict(attributes)[0]

# Write Predictions to Screen
st.subheader('The Predicted Number of Bicycle Rentals is: {0:.2f}'.format(prediction))