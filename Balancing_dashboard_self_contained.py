# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:00:33 2021

@author: ANDHM
"""

# cd C:\Users\ANDHM\Investment Analysis 3\Presentations
# streamlit run Balancing_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import pydeck as pdk
from functools import partial
import os

frequency_conversion = {'D':        1,
                        'W':        7,
                        'M':        30,
                        'SM':       15,
                        'Q':        91,
                        'SY':       182,
                        'Y':        365,
                        'SA':       182,
                        'A':        365,
                        'H':        1/24,
                        'T':        1/1440,
                        'min':      1/1440}

def get_duration(df: pd.DataFrame):
    delta_seconds = np.median(np.diff((df.index)))
    if isinstance(delta_seconds, pd.Timedelta):
        delta_seconds = delta_seconds.value/1e9
    else:
        delta_seconds = delta_seconds.item()/1e9
    return delta_seconds


def window_length(frequency: str, arg):
    string = [s for s in frequency if s.isdigit()]
    multiple = ''
    for s in string:
        multiple = multiple + s

    if string:
        multiple = int(multiple)
        unit = frequency.replace(str(multiple), '')
    else:
        multiple = 1
        unit = frequency

    if isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series):
        arg = obs_per_hour(arg)
    wdw = round(multiple * frequency_conversion[unit]) * 24 * arg
        
    return wdw
    
def obs_per_hour(df: pd.DataFrame):
    delta_seconds = get_duration(df)
    return round(3600/delta_seconds)
    
def scientific_notation(x):
    if not isinstance(x, float):
        x = x.squeeze()
    sign = int(np.sign(x))
    x = np.abs(x)
    
    power = int(np.floor(np.log10(x)))
    coeff = int(np.ceil(x/(10**power)))
    
    return str(sign).replace('1','')+str(coeff)+'e'+str(power)
    
def drop_trailing_nans(df: pd.DataFrame, drop='any'):
    if drop=='any':
        df = df[np.argmax(np.any(~np.isnan(df), axis=1)):]
        df = df[:len(df)-np.argmax(np.any(~np.isnan(df.iloc[::-1]), axis=1))]
    if drop=='all':
        df = df[np.argmax(np.all(~np.isnan(df), axis=1)):]
        df = df[:len(df)-np.argmax(np.all(~np.isnan(df.iloc[::-1]), axis=1))]
    return df

PATH_FILE = os.path.abspath(os.path.dirname(__file__))

class Balancing():    
    def __init__(self, forecast, actual, day_ahead, *imbalance_prices):
        self.over_production = actual-forecast
        self.actual = actual.rename()
        self.factor = obs_per_hour(self.over_production)
        
        if len(imbalance_prices)>1:
            day_ahead = day_ahead.rename()
            price_from_producer_to_system = imbalance_prices[0].rename()
            price_from_system_to_producer = imbalance_prices[1].rename()
            is_nan = price_from_producer_to_system.isna().__or__(price_from_system_to_producer.isna())
            assert np.mean(price_from_producer_to_system[~is_nan]<=price_from_system_to_producer[~is_nan])>0.99, 'Error in system prices. Consider switching around'
            self.net_system_price = price_from_producer_to_system*(self.over_production>=0)+price_from_system_to_producer*(self.over_production<0)-day_ahead
            self.net_system_price[np.isnan(self.over_production)] = np.NaN
        else:
            imbalance_prices = imbalance_prices[0]
            self.net_system_price = imbalance_prices-day_ahead
        
    def historical_cost(self, func, **kwargs):
        if 'frequency' in kwargs:
            wdw =  window_length(kwargs['frequency'], self.factor)
            del kwargs['frequency']
            kwargs['window'] = wdw
            
        if 'min_periods' in kwargs and isinstance(kwargs['min_periods'], str):
            min_periods =  window_length(kwargs['min_periods'], self.factor)
            kwargs['min_periods'] = min_periods
            
        func = partial(func, **kwargs)
        
        cost = -self.over_production*(self.net_system_price)
        balancing = func(cost).sum()/func(self.actual).sum()
        
        return balancing

    def decomposition(self, func, **kwargs):
        if 'frequency' in kwargs:
            wdw =  window_length( kwargs['frequency'], self.factor)
            del kwargs['frequency']
            kwargs['window'] = wdw
            
        if 'min_periods' in kwargs and isinstance(kwargs['min_periods'], str):
            min_periods =  window_length(kwargs['min_periods'], self.factor)
            kwargs['min_periods'] = min_periods
            
        func = partial(func, **kwargs)           
        
        mean_act = func(self.actual).mean()
        mean_Q = func(self.over_production).mean()
        mean_P = func(self.net_system_price).mean()
        mean_Q_sq = func(np.square(self.over_production)).mean()
        mean_P_sq = func(np.square(self.net_system_price)).mean()
        mean_QP = func(self.net_system_price * self.over_production).mean()

        std_Q = np.sqrt(mean_Q_sq-np.square(mean_Q))
        std_P = np.sqrt(mean_P_sq-np.square(mean_P))

        cov_QP = mean_QP-mean_Q*mean_P
        corr_QP = cov_QP/(std_Q*std_P)
        
        df_decomposition = pd.DataFrame(columns = ['std_Q', 'std_P', 'corr_QP', 'bias', 'forecast_error', 'basis'])
        df_decomposition.std_Q = std_Q/mean_act
        df_decomposition.std_P = std_P
        df_decomposition.corr_QP = corr_QP
        df_decomposition.bias = mean_Q*mean_P/mean_act
        df_decomposition.forecast_error = mean_Q/mean_act
        df_decomposition.basis = mean_P
        
        return df_decomposition
        
    def quote(self, timeline):
        std_Q = self.std_Q[-1]
        std_P = self.std_P[-1]
        corr_QP = self.corr_QP[-1]
        
        duration = (timeline-self.start_time).days/365
        
        q = std_Q*duration*self.Q_inc
        p = std_P*duration*self.P_inc
        qp = corr_QP*duration*self.corr_inc
        
        return q*p*qp 

def load_decomposition(country):
    path = os.path.join(PATH_FILE, r'balancing_{country}.hdf'.format(country = country))
    df_decomposition = pd.read_hdf(path, mode='r')    
    return df_decomposition

#%% Define data loading function
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def custom_decomposition(filename):
    df_entsoe = pd.read_csv(filename, parse_dates=[0], index_col=0, header=[0,1])
    df_entsoe.sort_index(axis=1, inplace=True)
    df_entsoe.resample(resolution).mean()
    
    all_actuals = df_entsoe.columns.get_loc_level('Actual', level=1)[1]
    all_forecasts = df_entsoe.columns.get_loc_level('Forecast', level=1)[1]
    power_sources = all_actuals.join(all_forecasts, how='inner')
    
    balancing_objects = {source: Balancing(df_entsoe[source]['Forecast'], df_entsoe[source]['Actual'], df_entsoe['Price']['Day-ahead'], 
                                              df_entsoe['Price']['Long'], df_entsoe['Price']['Short']) for source in power_sources}
    
    sample = pd.DataFrame.rolling
    kwargs = dict(frequency='Y', min_periods='6M', win_type='triang')
    
    df_balancing = pd.DataFrame({(source, 'Balancing'): balancing.historical_cost(sample, **kwargs) for source, balancing in balancing_objects.items()})
    df_decomposition = pd.concat({source: balancing.decomposition(sample, **kwargs) for source, balancing in balancing_objects.items()}, axis=1)
    df_decomposition = pd.concat([df_decomposition, df_balancing], axis=1)
    df_decomposition.sort_index(axis=1, inplace=True)
    df_decomposition = drop_trailing_nans(df_decomposition)
    
    return df_decomposition

df_coordinates = pd.DataFrame({'FR': {'Latitude': 48.8566, 'Longitude': 2.3522},
                               'ES': {'Latitude': 40.4168, 'Longitude': -3.7038},
                               'DK_1': {'Latitude': 56.1629, 'Longitude': 10.2039},
                               'DK_2': {'Latitude': 55.6761, 'Longitude': 12.5683},
                               'NL': {'Latitude': 52.3676, 'Longitude': 4.9041},
                               'GB': {'Latitude': 51.5074, 'Longitude': -0.1278}}).T

#%% Set parameters
st.set_page_config(layout='wide')
st.title('ENTSOE balancing dashboard')
st.sidebar.header('Parameters for dashboard')
countries = st.sidebar.multiselect('Select countries', ['FR','ES','GB','DK_1','DK_2'], ['FR','GB'])
factors = st.sidebar.multiselect('Select factors', ['Balancing', 'corr_QP', 'std_Q', 'std_P', 'forecast_error'], ['Balancing','corr_QP'])
sources = st.sidebar.multiselect('Select sources', ['Solar', 'Wind Offshore', 'Wind Onshore'], ['Solar', 'Wind Offshore', 'Wind Onshore'])
resolution = st.sidebar.select_slider('Resolution', options=['15T', '30T', '1H', '2H', '3H', '1D', '1W', '1M'], value='1D')
filename = st.sidebar.file_uploader("Upload your own data as csv file",type=["csv"])
np.random.seed(1000)

#%% Load and compute results
decompositions = [load_decomposition(country) for country in countries]

if not filename is None:
    st.spinner('Computing decomposition')
    df_custom = custom_decomposition(filename)
    
    decompositions.append(df_custom)
    countries.append('Custom data')
    
    csv = convert_df(df_custom)
    st.sidebar.download_button(label="Download decomposition as CSV", data=csv, file_name='custom_decomposition.csv', mime='text/csv')

#%% Display results
if countries and factors:
    columns = st.columns(len(countries))   
    for country, df_decomposition, column in zip(countries, decompositions, columns):    
        column.header(country)
        df_plot = df_decomposition[[nm for nm in df_decomposition if nm[0] in sources]].swaplevel(axis=1).resample(resolution).mean()
        for factor in factors:
            column.subheader(factor)
            column.line_chart(df_plot[factor])

#%% Display 3D map
#Set parameters
st.sidebar.header('Parameters for 3D map')
countries = st.sidebar.multiselect('Select countries', ['FR','ES','GB','DK_1','DK_2'], ['FR','ES','GB','DK_1','DK_2'])
factor = st.sidebar.selectbox('Factor for 3D map', ('Balancing', 'std_Q', 'corr_QP', 'std_P', 'forecast_error'), 3)
source = st.sidebar.selectbox('Source for 3D map', ('Solar', 'Wind Onshore', 'Wind Offshore'), 1)
start_time = st.sidebar.slider("Display time", datetime(2016, 1, 1, 0), datetime(2021, 5, 5, 0), value=datetime(2021, 2, 11, 0), format="DD/MM/YY - hh")

#Compute summary data
decompositions = [load_decomposition(country) for country in countries]
imbalance_costs = pd.Series({country: df[source][factor][start_time] if source in df.columns.get_level_values(0) else None for country, df in zip(countries, decompositions)})
imbalance_costs = imbalance_costs.dropna()
imbalance_costs = imbalance_costs.apply(lambda x: int(scientific_notation(np.abs(x))[0])*100)
df = pd.concat([df_coordinates.loc[[country]*imbalance_costs[country]] for country in imbalance_costs.index], axis=0)

#Control parameters for deck chart
area = (df_coordinates.loc[countries].max(axis=0)-df_coordinates.loc[countries].min(axis=0)).prod()
zoom = 6/np.power(area,0.1)
radius = 5000*np.power(area,0.3)
elevation_max = 10000*np.power(area, 0.3)
elevation_scale = 4

#Inputs for deck chart
view = pdk.ViewState(latitude=df['Latitude'].mean(), longitude=df['Longitude'].mean(), zoom=zoom, pitch=60)
hex_layer = pdk.Layer('HexagonLayer', data=df, get_position='[Longitude, Latitude]', radius=radius, elevation_scale=elevation_scale, 
                      elevation_range=[0, elevation_max], pickable=True, extruded=True)
scatter_layer = pdk.Layer('ScatterplotLayer', data=df, get_position='[Longitude, Latitude]', get_color='[200, 30, 0, 160]', get_radius=radius)

#Display deck chart
with st.expander("3D map"):
    st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9', initial_view_state=view, layers=[hex_layer, scatter_layer]))