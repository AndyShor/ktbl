# ktbl - data aqusition and ML analysis of kettlebell exercises

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/license/gpl-3-0) 

![cover](/reports/img/cover.png)


# About

KTBL project is an attempt to make training with kettlebells more efficient and fun.
The contained software allows to record and transmitt IMU data provided by a battery powered sensor,
attached to a kettlebell. On the host PC the data is saved and can be analyzed using pretrained CNN and
DTW technics in a post processor. Presently we focus on identifying and counting classical kettlebell moves such as swing, snatch and jerk (from chest).

Currently data aqusition is performed using Arduino Nano Sense (Rev 1 or 2).
Model is trained and reference data for DTW distance calculation is recorded by the author himself.
Model is tested on volunteers with varying degrees of general fitness, age and body size. In the field test all volunteers used the same 12 kg kettlebell.  



# How to use

## What is included

### daq 
 data aquistion from IMU and data streaming realized by a BLE peripheral device (Arduino nano sense)
 data logging by a BLE central device realized using Python bleak library (Win/Linux/Mac)

 ### data 
raw - recorded time series data as received
interim - label dictionaries
final - 1.6 seconds labeled data segments, interpolated, smoothed, equally spaced, ready to be fed to a CNN model

 ### models
best trained model

 ### notebooks
data exploration notebook
data transformation notebook
CNN model training notebook
DTW estimation notebook

 ### postprocessing
streamlit app that ingests time series data and shows CNN, FFt and DTW analysis of the data

 ### reports
contains experimental findings during tests on volunteers





