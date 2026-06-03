#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:06:52 2024

@author: fionaspuler
"""

import numpy as np
import xarray as xr
import pandas as pd


def filter_morocco(dataset):
    return(dataset.sel(latitude=slice(36,30), longitude=slice(-11,0)))

def filter_cca_region(dataset):
    return(dataset.sel(latitude=slice(20,50), longitude=slice(-30,20)))

def filter_mediterranean(dataset):
    return(dataset.sel(latitude=slice(25, 50), longitude=slice(-20,45)))

def filter_smaller_mediterranean(dataset):
    return(dataset.sel(latitude=slice(30, 50), longitude=slice(-20,40)))

def filter_larger_mediterranean(dataset):
    return(dataset.sel(latitude=slice(25, 60), longitude=slice(-30,45)))

def filter_atlantic(dataset):
    return(dataset.sel(latitude=slice(25, 80), longitude=slice(-50,30)))

def filter_north_atlantic(dataset):
    return(dataset.sel(latitude=slice(50, 65), longitude=slice(-60,0)))

def filter_new_atlantic(dataset):
    return(dataset.sel(latitude=slice(20, 80), longitude=slice(-50,30)))

def calculate_anomalies(x):
    return(x-x.mean(dim='time'))


def preprocess_dataset(filename, variable_name, multiplication_factor, 
                       geographical_filter, months_filter, anomalies, normalization,
                       rolling_window):
    
    dataset = xr.open_dataset(filename)[variable_name]*multiplication_factor
    
    if geographical_filter=='mediterranean':
        dataset = filter_mediterranean(dataset)
    elif geographical_filter=='morocco':
        dataset = filter_morocco(dataset)
    elif geographical_filter=='larger mediterranean':
        dataset = filter_larger_mediterranean(dataset)
    elif geographical_filter=='atlantic':
        dataset = filter_atlantic(dataset)
    elif geographical_filter=='north atlantic':
        dataset = filter_north_atlantic(dataset)
        
    elif geographical_filter=='cca':
        dataset = filter_cca_region(dataset)
        
    elif geographical_filter=='new atlantic':
            dataset = filter_new_atlantic(dataset)
    else:
        print('Geographical filter not recognized, no filter applied')
        
    dataset = dataset.sel(time=np.isin(dataset.time.dt.month, months_filter))
    
    if anomalies==True:
        dataset = dataset.groupby('time.dayofyear').map(calculate_anomalies)
    if normalization==True:
        dataset = dataset/dataset.std(dim='time')
    if rolling_window!=0:
        dataset = dataset.rolling(time=rolling_window, min_periods=1, center=True).mean()
        
    return(dataset)