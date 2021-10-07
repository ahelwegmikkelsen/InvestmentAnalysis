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
from balancing_package import load_decomposition, Balancing
from utils import drop_trailing_nans, convert_to_frequency, scientific_notation
import time
from datetime import datetime, timezone
import pydeck as pdk

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