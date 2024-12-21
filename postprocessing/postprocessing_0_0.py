import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
#import tensorflow as tf
from tensorflow import keras
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from dtaidistance import dtw


st.title("Timeseries data post processing ")
span = 1.6 # span of analysys
p_number=50
# load best trained model
model = keras.models.load_model("best_model.h5")

uploaded_file = st.file_uploader("Upload data file", type={"csv"})

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file, names=['time','gFx', 'gFy', 'gFz', 'TgF'], skiprows=1)
    uploaded_file.close()
    #st.write(uploaded_df.describe())
    st.markdown(f"Quick post process interface for **Time series data** ")

    fig1, ax1 = plt.subplots()
    plt.plot(uploaded_df['time'], uploaded_df['gFz'], label='x')

    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')
    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('value', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')
    st.pyplot(fig1)


    #st.line_chart(uploaded_df.gFx)
    #t_start = uploaded_df['time'].values[0]
    t_start=uploaded_df['time'].values[0]+span/2+0.2
    #t_stop = uploaded_df['time'].values[-1]
    t_stop=max(uploaded_df['time'].values)-span/2-0.2
    #t_stop = 17
    t_steps = np.linspace(t_start, t_stop, num=int((t_stop - t_start) * 4))
    xvals = np.linspace(0, span, p_number)

    jerk_dist = np.empty(len(t_steps))
    swing_dist = np.empty(len(t_steps))
    snatch_dist = np.empty(len(t_steps))
    none_dist = np.empty(len(t_steps))
    noswing_dist = np.empty(len(t_steps))
    total_prob = np.empty(len(t_steps))

    for idx, timestep in enumerate(t_steps):
        test_ts = uploaded_df[(uploaded_df['time'] > timestep - span/2) & (uploaded_df['time'] < timestep + span/2)]
        start_time = test_ts['time'].values[0]
        sliding_rel_time = test_ts['time'].values - start_time
        sliding_gFy = np.interp(xvals, sliding_rel_time, test_ts['gFy'].values)
        sliding_gFz = np.interp(xvals, sliding_rel_time, test_ts['gFz'].values)
        #sliding_TgF = np.interp(xvals, sliding_rel_time, test_ts['TgF'].values)



        sliding_gF2d = sliding_gFy.tolist() + sliding_gFz.tolist()
        x_sliding = [sliding_gF2d]
        x_sliding = np.asarray(x_sliding).astype(np.float32)
        predictions = model.predict(x_sliding, verbose=0)
        jerk_dist[idx] = predictions[0][1]
        swing_dist[idx] = predictions[0][0]
        snatch_dist[idx] = predictions[0][2]
        none_dist[idx] = predictions[0][3]
        noswing_dist[idx] = predictions[0][4]
        total_prob[idx] = jerk_dist[idx] + swing_dist[idx] + snatch_dist[idx] + noswing_dist[idx]
        #print('step completed')
        #st.markdown(f" step completed ")

    # plotting inference results
    fig, ax = plt.subplots()
    plt.plot(t_steps, jerk_dist, label='jerk')
    plt.plot(t_steps, swing_dist, label='swing')
    plt.plot(t_steps, snatch_dist, label='snatch')
    plt.plot(t_steps, none_dist, label='none')
    #plt.plot(t_steps, noswing_dist, label='noswing')
    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')
    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('event probability', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')

    st.markdown(f" CNN data analysys ")
    st.pyplot(fig)

    # FFT data analysys
    # number of signal points
    N = len(uploaded_df['TgF'].values)
    T=(max(uploaded_df['time'].values)-min(uploaded_df['time'].values))/N
    fft_data_z= fft(uploaded_df['gFz'].values)
    fft_data_y = fft(uploaded_df['gFy'].values)
    fft_x=fftfreq(N, T)[:N//2]
    fig_fft, ax_fft = plt.subplots()
    plt.plot(fft_x[2:], 2.0/N * np.abs(fft_data_z[2:N//2]), label='FFT z')
    plt.plot(fft_x[2:], 2.0 / N * np.abs(fft_data_y[2:N // 2]), label='FFT y')
    plt.xlabel('frequency [Hz]')
    #plt.xlim((0,20))
    plt.xlim((0, 20))
    plt.ylabel('FFT')
    st.pyplot(fig_fft)
    # DTW data analysys
    mean_swing=pd.read_csv('mean_swing.csv')
    mean_jerk = pd.read_csv('mean_jerk.csv')
    mean_snatch = pd.read_csv('mean_snatch.csv')
    jerk_dist = np.empty(len(t_steps))
    swing_dist = np.empty(len(t_steps))
    snatch_dist = np.empty(len(t_steps))
    xvals = np.linspace(0, span, p_number)
    a=0.1 # regularizing parameter for inverted distance

    for idx, timestep in enumerate(t_steps):
        test_ts = uploaded_df[(uploaded_df['time'] > timestep - span/2) & (uploaded_df['time'] < timestep + span/2)]
        start_time = test_ts['time'].values[0]
        sliding_rel_time = test_ts['time'].values - start_time
        sliding_gFy = np.interp(xvals, sliding_rel_time, test_ts['gFy'].values)
        sliding_gFz = np.interp(xvals, sliding_rel_time, test_ts['gFz'].values)

        jerk_dist[idx] = dtw.distance_fast(sliding_gFy, mean_jerk['gFy'].values) + dtw.distance_fast(sliding_gFz, mean_jerk['gFz'].values)
        swing_dist[idx] = dtw.distance_fast(sliding_gFy, mean_swing['gFy'].values) + dtw.distance_fast(sliding_gFz,mean_swing['gFz'].values)
        snatch_dist[idx] = dtw.distance_fast(sliding_gFy, mean_snatch['gFy'].values) + dtw.distance_fast(sliding_gFz, mean_snatch['gFz'].values)


    # plt.plot(mean_jerk['rel_t'],mean_jerk['TgF'], label='TgF')



    fig_dtw, ax_dtw = plt.subplots()
    plt.plot(t_steps, 1 / ((jerk_dist) ** 2 + a), label='jerk')
    plt.plot(t_steps, 1 / ((swing_dist) ** 2 + a), label='swing')
    plt.plot(t_steps, 1 / ((snatch_dist) ** 2 + a), label='snatch')

    plt.xlabel('time [s]', fontsize=15)
    plt.ylabel('inverted DTW distance', fontsize=15)
    plt.style.use('classic')
    # plt.yscale("log")
    plt.legend(loc='upper right')

    st.pyplot(fig_dtw)












