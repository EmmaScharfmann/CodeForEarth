#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:06:52 2024

@author: fionaspuler
"""

import numpy as np
import xarray as xr
import pandas as pd

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy

def reshape_data_for_clustering(xarray_data):
    
    # extract numpy array from xarray object
    data = xarray_data.values
    
    # reshape
    nt,ny,nx = data.shape
    data = np.reshape(data, [nt, ny*nx], order='F')
    
    return(data)

def calculate_regime_length(labels):

    j=1
    l=labels[0]

    lengths = []

    for i in range(len(labels)-1):

        if(labels[i+1]==labels[i]):
            j=j+1

        else:
            lengths.append(pd.DataFrame(data={"Regime": l,"Length": j}, index=[0]))
            l = labels[i+1]
            j=1

    lengths_df = pd.concat(lengths) 
    return(lengths_df)  



def calculate_conditional_probability_change_label(threshold_matrix, labels, comparison, shift_value=0):
    
    # add cluster assignment to threshold vector
    threshold_matrix_label = threshold_matrix.assign_coords(label=("time", np.roll(labels, shift_value)))

    # probability conditional on weather type
    n_wr = threshold_matrix_label.groupby('label').mean()

    # overall probability
    n_total = threshold_matrix_label.mean(dim='time')
    
    if comparison=='difference':
        ds = n_wr - n_total
    elif comparison=='ratio':
        ds = n_wr/n_total
    elif comparison=='none':
        ds = n_wr
    else:
        print('invalid entry for diff_or_quot')
    
    return(ds)



def visualise_contourplot(cluster_centers, unit, regime_names, vmin, vmax, steps, color_scheme, 
                                   labels_data, labels, col_number=8, borders=True, projection = ccrs.Orthographic(0,45)):
    
    nt,ny,nx = cluster_centers.values.shape
    x,y = np.meshgrid(cluster_centers.longitude, cluster_centers.latitude)
    
    proj = projection
    fig, axes = plt.subplots(1,col_number, figsize=(14, 5), subplot_kw=dict(projection=proj))

    regimes = regime_names

    for i in range(nt):
        
        cs = axes.flat[i].contourf(x, y, cluster_centers[i, :, :],
                                   levels=np.arange(vmin, vmax, steps), 
                                   transform=ccrs.PlateCarree(),
                                   cmap=color_scheme)
        axes.flat[i].coastlines()
        
        if borders==True:
            axes.flat[i].add_feature(cartopy.feature.BORDERS)
            
        title = '{}, {:4.1f}%'.format(regimes[i], 100*labels_data[labels==i, :].shape[0]/labels_data.shape[0])
        axes.flat[i].set_title(title)
    plt.tight_layout()
    
    return(fig)





def visualise_spatial_oddsratio(dataset_xarray, unit, color_scheme, vmin, vmax, steps, 
                                title, regime_names, borders=True, projection=ccrs.PlateCarree(central_longitude=0), col_number=8):
    
    nt,ny,nx = dataset_xarray.values.shape
    
    proj=projection
    
    regimes = regime_names
    
    fig, axes = plt.subplots(1,col_number, figsize=(14, 5), subplot_kw=dict(projection=proj))
    
    for i in range(nt):
        
        cs = dataset_xarray[i, :, :].plot(ax=axes.flat[i], colors=color_scheme,
                                          transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, levels=steps,
                                         add_colorbar=False)
        axes.flat[i].coastlines()
        
        if borders==True:
            axes.flat[i].add_feature(cartopy.feature.BORDERS)
            
        sub_title = '{}'.format(regimes[i])
        axes.flat[i].set_title(sub_title)

    plt.tight_layout()
    return(fig)