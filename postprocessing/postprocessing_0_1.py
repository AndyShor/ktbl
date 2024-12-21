import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
#from scipy.signal import savgol_filter
#import tensorflow as tf
from tensorflow import keras
#from scipy import signal
from scipy.fft import fft, fftfreq
from dtaidistance import dtw
from scipy.signal import find_peaks
import os


st.title("Timeseries data post processing ")
span = 1.6 # span of analysys
p_number=50 # numbe, secondsr of points per span
window_step=0.2 # step of the window, seconds
win_n_steps=int(p_number*window_step/span) # number of steps in one window shift

def read_gf_data(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        gFy_idx = header.index('gFy')
        gFz_idx = header.index('gFz')
        data = np.genfromtxt(f, delimiter=',', skip_header=0, usecols=(gFy_idx, gFz_idx))
    gFy_array = data[:, 0]
    gFz_array = data[:, 1]
    return gFy_array, gFz_array

# load average motions for DTW comparison

#swing_gFy, swing_gFz = read_gf_data('mean_swing.csv')
#jerk_gFy, jerk_gFz = read_gf_data('mean_jerk.csv')
#snatch_gFy, snatch_gFz = read_gf_data('mean_snatch.csv')

mean_swing=pd.read_csv('mean_swing.csv')
mean_jerk = pd.read_csv('mean_jerk.csv')
mean_snatch = pd.read_csv('mean_snatch.csv')
swing_gFy=mean_swing['gFy'].values
swing_gFz=mean_swing['gFz'].values
jerk_gFy=mean_jerk['gFy'].values
jerk_gFz=mean_jerk['gFz'].values    
snatch_gFy=mean_snatch['gFy'].values    
snatch_gFz=mean_snatch['gFz'].values



# Construct the path to the model file

model_path = '../models/best_model_2f.h5'

# Load the pre-trained model
model = keras.models.load_model(model_path)

uploaded_file = st.file_uploader("Upload data file", type={"csv"})

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file, names=['time','gFx', 'gFy', 'gFz', 'TgF'], skiprows=1)
    uploaded_file.close()
    #st.write(uploaded_df.describe())
    t_min = uploaded_df['time'].values[0]
    t_max= uploaded_df['time'].values[-1]
    data_timespan= t_max - t_min
    st.markdown(f"Quick post process interface for **Time series data** ")
    st.markdown(f" time span of time series data is {data_timespan:.2f} seconds ")
    # calculate total number of steps from length, span, p_number
    total_n_steps=int(p_number*data_timespan/span)

    # create time scale for raw data interpolation
    t_scale= np.linspace(t_min, total_n_steps*span/p_number, total_n_steps)
    # interpolate raw data for uniformity of time spacing
    gFy_interp= np.interp(t_scale, uploaded_df['time'].values, uploaded_df['gFy'].values)
    gFz_interp = np.interp(t_scale, uploaded_df['time'].values, uploaded_df['gFz'].values)
    # create rolling window views accelerations
    gFy_rolling = np.lib.stride_tricks.sliding_window_view(gFy_interp, p_number)
    gFz_rolling = np.lib.stride_tricks.sliding_window_view(gFz_interp, p_number)
    # compress views to have window slide with a number of steps defined by window step variable
    gFy_rolling_compressed=gFy_rolling[0:-1:win_n_steps][:]
    gFz_rolling_compressed = gFz_rolling[0:-1:win_n_steps][:]
    # concatenate two accelerations into one array for CNN data inference
    cnn_inference=np.concatenate([gFy_rolling_compressed,gFz_rolling_compressed], axis=1 )
    # infer the data
    predictions = model.predict(cnn_inference, verbose=0)
    # save inference results in separate arrays
    swing_dist = predictions[:, 0]
    jerk_dist = predictions[:,1]
    snatch_dist = predictions[:,2]
    none_dist = predictions[:,3]
    noswing_dist = predictions[:,4]
    # generate time scale of CNN results, time is shorter than raw data by the window size
    t_plot=np.linspace(0, t_max-t_min-span, len(predictions) )

    print(len(predictions))
    print(len(gFy_rolling_compressed))

    #swing_dist_dtw=[dtw.distance_fast(x, mean_swing['gFy'].values) for x in gFy_rolling_compressed ]
    #print(len(swing_dist_dtw))


    # FFT data analysys
    # number of signal points
    N = len(uploaded_df['TgF'].values)
    T=data_timespan/N
    fft_data_z = fft(uploaded_df['gFz'].values)
    fft_data_y = fft(uploaded_df['gFy'].values)
    fft_x = fftfreq(N, T)[:N // 2]


    # DTW data analysys
    #swing_dist_dtw=np.array([dtw.distance_fast(row.copy(), swing_gFy, use_c=True) for row in gFy_rolling_compressed])
    a = 0.1  # regularizing parameter for inverted distance

    swing_dist_dtw= np.array([1/((dtw.distance_fast(gFy_rolling_compressed[i].copy(), swing_gFy) + dtw.distance_fast(gFz_rolling_compressed[i].copy(), swing_gFz) )**2+a) for i in range(gFy_rolling_compressed.shape[0])])
    jerk_dist_dtw=np.array([1/((dtw.distance_fast(gFy_rolling_compressed[i].copy(), jerk_gFy) + dtw.distance_fast(gFz_rolling_compressed[i].copy(), jerk_gFz) )**2+a) for i in range(gFy_rolling_compressed.shape[0])])
    snatch_dist_dtw= np.array([1/((dtw.distance_fast(gFy_rolling_compressed[i].copy(), snatch_gFy) + dtw.distance_fast(gFz_rolling_compressed[i].copy(), snatch_gFz) )**2+a) for i in range(gFy_rolling_compressed.shape[0])])
   

    
    

    def extract_features(swing_dist_dtw, jerk_dist_dtw, snatch_dist_dtw, N):
        # Find peaks in each time series
        swing_peaks, swing_props = find_peaks(swing_dist_dtw, distance=int(1.3/window_step), prominence=0.04)
        jerk_peaks, jerk_props = find_peaks(jerk_dist_dtw, distance=int(1.3 / window_step), prominence=0.04)
        snatch_peaks, snatch_props = find_peaks(snatch_dist_dtw, distance=int(1.3 / window_step), prominence=0.04)

        # Initialize list to store features
        features = []

        # Loop over peak locations
        for peak_index in np.unique(np.concatenate([swing_peaks, jerk_peaks, snatch_peaks])):
            # Find peaks within time span from indexes peak_index-N to peak_index+N
            mask = (swing_peaks >= peak_index-N) & (swing_peaks <= peak_index+N)
            swing_max_prominence = np.max(swing_props['prominences'][mask]) if np.any(mask) else 0

            mask = (jerk_peaks >= peak_index-N) & (jerk_peaks <= peak_index+N)
            jerk_max_prominence = np.max(jerk_props['prominences'][mask]) if np.any(mask) else 0

            mask = (snatch_peaks >= peak_index-N) & (snatch_peaks <= peak_index+N)
            snatch_max_prominence = np.max(snatch_props['prominences'][mask]) if np.any(mask) else 0

            # Find peak with highest prominence
            max_prominence = np.max([swing_max_prominence, jerk_max_prominence, snatch_max_prominence])
            if max_prominence > 0:
                if  max_prominence == swing_max_prominence and peak_index in swing_peaks:
                    features.append({
                        'index': peak_index,
                        'type': 'swing',
                        'prom': max_prominence
                    })
                elif max_prominence == jerk_max_prominence and peak_index in jerk_peaks:
                    features.append({
                        'index': peak_index,
                        'type': 'jerk',
                        'prom': max_prominence
                    })
                elif max_prominence == snatch_max_prominence and peak_index in snatch_peaks:
                    features.append({
                        'index': peak_index,
                        'type': 'snatch',
                        'prom': max_prominence
                    })
                  


        return features


    
    features = extract_features(swing_dist_dtw, jerk_dist_dtw, snatch_dist_dtw, int(1.3 / window_step))
    #print(features)


    # plot FFT data
    fig_fft, ax_fft = plt.subplots()
    plt.plot(fft_x[2:], 2.0 / N * np.abs(fft_data_z[2:N // 2]), label='FFT z')
    plt.plot(fft_x[2:], 2.0 / N * np.abs(fft_data_y[2:N // 2]), label='FFT y')
    plt.xlabel('frequency [Hz]')
    # plt.xlim((0,20))
    plt.xlim((0, 20))
    plt.ylabel('FFT')
    plt.legend(loc='upper right')
    st.pyplot(fig_fft)


   # plot total acceleration from raw data

    fig1, ax1 = plt.subplots()
    plt.plot(uploaded_df['time'], uploaded_df['TgF'], label='TgF')
    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')
    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('TgF', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')
    st.pyplot(fig1)


    # plotting CNN inference results
    fig, ax = plt.subplots()
    plt.plot(t_plot, jerk_dist, label='jerk')
    plt.plot(t_plot, swing_dist, label='swing')
    plt.plot(t_plot, snatch_dist, label='snatch')
    plt.plot(t_plot, none_dist, label='none')
    #plt.plot(t_steps, noswing_dist, label='noswing')
    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')
    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('event probability', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')

    st.markdown(f" CNN data analysys ")
    st.pyplot(fig)

    
    
    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')


    color_dict={"jerk":"b","swing":"g","snatch":"r","none":"y"}
    fig_dtw, ax_dtw = plt.subplots()
    plt.plot(t_plot, jerk_dist_dtw, label='jerk')
    plt.plot(t_plot,swing_dist_dtw, label='swing')
    plt.plot(t_plot, snatch_dist_dtw, label='snatch')
    #swing_lines=[plt.axvline(t_plot[p], c='g', linewidth=0.3) for p in swing_peaks]
    #jerk_lines = [plt.axvline(t_plot[p], c='b', linewidth=0.3) for p in jerk_peaks]

    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('inverted DTW distance', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')

    # Loop over the features
    for feature in features:
        # Get the x-coordinate from the time array
        x = t_plot[feature['index']]

        # Get the y-coordinate from the feature prominence
        y = feature['prom']

        # Get the color from the feature type
        color = color_dict[feature['type']]

        # Add a small circle to the plot
        plt.plot(x, y, 'o', markersize=5, markeredgecolor=color, markerfacecolor=color)


    st.pyplot(fig_dtw)


    













