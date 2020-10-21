from datetime import timedelta
from os import path
from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from fbprophet import Prophet
import streamlit
from datetime import datetime
import os
import numpy as np
st.title('Covid prediction - D3 Test')
st.subheader('Prediction test using linear regression and time series prediction')

DATA_URL = ('https://covid.ourworldindata.org/data/owid-covid-data.csv')

# Components
CHOOSE_ALGORITHM = ['Time Series', 'Linear Regression']
next_days = st.slider('Select the number of future days', min_value=1, max_value=365, value=10)
algorithm_type_choose = st.selectbox('Algorithm used', CHOOSE_ALGORITHM)

@streamlit.cache(allow_output_mutation=True)
def load_data()->DataFrame:
    """
    Delete all files where downloaded in the past
    """
    ## check directory
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
    now = datetime.now()
    file_name = f'{now.strftime("%Y-%m-%d")}.csv'
    files_to_remove = os.listdir('./data')
    
    # filter and delete olders
    files_to_remove = [os.path.join('./data',f) for f in files_to_remove if f != file_name]
    for f in files_to_remove:
        os.remove(f)
    
    # join path
    file_name = os.path.join('./data', file_name)
    
    # check if not exists make download
    if not os.path.exists(file_name):
        r = requests.get(DATA_URL)  
        with open(file_name, 'wb') as f:
            f.write(r.content)
    
    #read the data
    original_data = pd.read_csv(file_name)
    return original_data

@streamlit.cache(allow_output_mutation=True)   
def filter_only_world(original_data:DataFrame)->DataFrame:
    filtered = original_data[original_data['location'] == 'World'].copy()
    filtered.reset_index(drop=False, inplace=True)
    filtered['day'] = range(1, len(filtered) + 1)
    filtered = filtered[['day', 'total_cases', 'date']].copy()
    return filtered

def fit_linear_regression(df:DataFrame, next_days:int, portion_size_to_train:int)->DataFrame:
    # linear regression
    new_data_frame = df.copy()
    lm = LinearRegression(fit_intercept=True)
    new_data_frame.reset_index()
    #Convert date to datetime
    new_data_frame['date'] = pd.to_datetime(new_data_frame['date'])

    X = new_data_frame['day'].to_numpy().reshape(-1,1)
    y = new_data_frame['total_cases'].to_numpy().reshape(-1,1)

    # define what days is used to train model
    x_train = X[-portion_size_to_train:] if first_or_last_days else X[:portion_size_to_train]
    y_train = y[-portion_size_to_train:] if first_or_last_days else y[:portion_size_to_train]

    lm.fit(x_train, y_train)
    
    # find the last row
    last_row = new_data_frame.iloc[[-1]]

    # add the nexts days to predict
    for x in range(1,next_days+1):
        row = dict(
            day=int(last_row['day']) + x,
            date=last_row['date'] + timedelta(days=x),
            total_cases = 'nan',
        )
        new_data_frame = new_data_frame.append(row, ignore_index=True)
    
    # linear regression
    if algorithm_type_choose == CHOOSE_ALGORITHM[1]:
        new_data_frame['linear_regression_prediction'] = lm.predict(new_data_frame['day'].to_numpy().reshape(-1,1)).astype('int')

    ## time series
    if algorithm_type_choose == CHOOSE_ALGORITHM[0]:
        time_df = df.copy()
        time_df.rename(columns={'date': 'ds','total_cases':'y' }, inplace=True)
        m = Prophet()
        time_df_test = time_df[-portion_size_to_train:] if first_or_last_days else time_df[:portion_size_to_train]
        m.fit(time_df_test)
        future = m.make_future_dataframe(periods=next_days)
        forecast = m.predict(future)
        new_data_frame['time_series_prediction'] = forecast['yhat'].astype('int')
        #new_data_frame['time_series_prediction'].apply(lambda x : "{:,}".format(x))
    
    # remove the column date
    del new_data_frame['date']
    
    
    # filter the table predictions
    predictions = new_data_frame[new_data_frame['day'] > int(last_row['day'])].copy()
    del predictions['total_cases']

    return new_data_frame, predictions
   

data_load_state = st.text('Loading data...')
original_data = load_data()


data_load_state.text('Data loaded ...')


filtered = filter_only_world(original_data)
first_or_last_days = st.checkbox('Use the last days to older days', True)
slide_portion = st.slider('Portion days used to train the model', min_value=30, max_value=len(filtered), value=len(filtered))
st.subheader('Original data')
st.write(filtered)
data, prediction = fit_linear_regression(filtered, next_days, slide_portion)

st.subheader('Number of cases per day predicted')
st.write(prediction)
st.subheader('Prediction growth')

st.line_chart(data)
