# Machine Learning Internship at FTS
[Project Report](./FTS.pdf)
## Air Quality Index Prediction
As air pollution is a complex mixture of toxic components with considerable impact on humans, forecasting air pollution concentration emerges as a priority for improving life quality. So with the help of Python tools and some Machine Learning algorithms, we try to predict the air quality.

### Introduction
During the project We were given two datasets:

    1. cities\_by\_day → day-wise information including the amount of various chemical substances present in different cities and the AQI information.
    2. cities\_by\_hours → hours-wise information including the amount of various chemical substances present in different cities and the AQI information.
### Dataset information

    The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value. This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

### Attribute information
    - 0 Date	(DD/MM/YYYY) 
    - 1 Time	(HH.MM.SS) 
    - 2 True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
    - 3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)	
    - 4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
    - 5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
    - 6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
    - 7 True hourly averaged NOx concentration in ppb (reference analyzer) 
    - 8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
    - 9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
    - 10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
    - 11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
    - 12 Temperature in Â°C	
    - 13 Relative Humidity (%) 
    - 14 AH Absolute Humidity 


### Install
This project requires Python 3.6 and the following libraries installed:
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)
- [Tensorflow](https://www.tensorflow.org/)

### Approach For Analysing Data
 * We have initially performed Exploratory Data Analysis including Data preprocessing, Outlier.
 * treatment and Data visualization to study the datasets.
 * We have then used certain algorithms like XGBoost and Stacked LSTM to create a model that
 * will predict the AQI for any future reference using the input we are giving.


## Models trained on
1. Linear Regression
2. Linear Regression with L1, L2 regularization
3. Boosted models
4. Stacked LSTM