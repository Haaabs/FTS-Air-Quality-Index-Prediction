
# Machine Learning Internship at FTS

## Air Quality Index Prediction

By,
Akshata Kotti
Shubham Urmaliya
Abhishek
Shreya Basu
Neha Kumari
Lokesh

## Contents
    - Introduction
    - Data Visualisation before preprocessing
    - Data Visualisation on AQI
    - Data preprocessing
    - Missing value treatment
    - Air Quality Index(AQI) calculation
    - Outlier treatment
    - Data Visualisation after preprocessing
    - Model Building
    - XGBoost
    - Stacked LSTM

### Introduction
During the project We were given two datasets:

    1. cities\_by\_day → day-wise information including the amount of various chemical substances present in different cities and the AQI information.
    2. cities\_by\_hours → hours-wise information including the amount of various chemical substances present in different cities and the AQI information.
### Approach For Analysing Data
 *  We have initially performed Exploratory Data Analysis including Data preprocessing, Outlier.
 * treatment and Data visualization to study the datasets.
 * We have then used certain algorithms like XGBoost and Stacked LSTM to create a model that
 * will predict the AQI for any future reference using the input we are giving.


### Data Visualisation before preprocessing
* We used some visualisation techniques to understand the trends and relationships between different
columns. The results are following.

There are a lot of missing values for xylene,PM2.5 and NH3. But after looking at correlations

AQI is reasonably dependent on these gases. So it is not good to drop these columns.

● The second image is a plot of PM(PM2.5 +PM10) with months. From this graph we can see that

values are not missing at random they are missing for long periods of time from this we found

that the imputation methods like linear interpolation will not give realistic results and we started

thinking about methods like KNN imputation.





**Data Visualisation on AQI**

Data visualization is the graphical representation of information and data. We use different [visual](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[elements](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[like](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[charts,](https://www.tableau.com/learn/articles/data-visualization/glossary)

[graphs,](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[and](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[ ](https://www.tableau.com/learn/articles/data-visualization/glossary)[maps](https://www.tableau.com/learn/articles/data-visualization/glossary), data visualization tools to provide an accessible way to see and understand trends, outliers, and

patterns in data.

Visualization has been done on the dataset of cities\_by\_day to study certain trends. Some screenshots have been

attached herewith. The link to the file has been given here:

<https://colab.research.google.com/drive/1UIySiXXD82j0ocehY9wtBLlZj7am7gl-#scrollTo=RHBP32Q3qcLu>

**Calculating the proportion**

**of missing values**

**Grouping the cities based on average AQI**

**Pie-chart showing distribution of**

**pollutant in top polluted cities**





**Data preprocessing**

**KNN Imputation**

**Outlier Detection Using Quantile Regression**

**def** fun(dframe):

lis = []

Q1=df['AQI\_calculated'].quantile(0.25)

Q3=df['AQI\_calculated'].quantile(0.75)

IQR=Q3-Q1

print(Q1)

print(Q3)

**for** i **in** range(0, dframe.shape[1]):

**if**(dframe.iloc[:,i].dtypes == 'object'):

dframe.iloc[:,i] = pd.Categorical(dframe.iloc[:,i])

dframe.iloc[:,i] = dframe.iloc[:,i].cat.codes

dframe.iloc[:,i] = dframe.iloc[:,i].astype('object')

print(IQR)

Lower\_Whisker = Q1 - 1.5\*IQR

Upper\_Whisker = Q3 + 1.5\*IQR

print(Lower\_Whisker, Upper\_Whisker)

df = df[df['AQI\_calculated']< Upper\_Whisker]

lis.append(dframe.columns[i])

KNN = KNNImputer(n\_neighbors=3)

dframe = pd.DataFrame(KNN.fit\_transform(dframe))

**return** dframe





**Data Preprocessing of Cities\_by\_day and Cities\_by\_hours dataset**

**1] Missing value treatment**: Methods used to treat missing values are:

● Citywise Mean imputation

● Citywise Linear interpolation

● Citywise K-Nearest Neighbors(KNN) imputation

**2] AQI calculation:** AQI is the maximum of sub-indices calculated for individual pollutants.

**3] Outlier treatment**: Outliers were detected and treated using Quantile Regression.

**Percentage of missing**

**values in cities\_by\_day:**

**Percentage of missing**

**values in cities\_by\_hour:**





**Data Visualisation after preprocessing**

Visualization has also been performed after preprocessing the dataset cities\_by\_hours i.e., removing the

missing values in the dataset.

**Proportion of missing**

**values has been reduced**

**to zero**

**Pie-chart showing imputed AQI**

**values for top polluted cities**

**Correlation analysis**





**Model Making - (i) XGBoost Regressor**

n\_estimators = [int(x) **for** x **in** np.linspace(start=100, stop=1200, num=12)]

**def** fun(Ahm):

Ahm.drop(['City'],axis=1,inplace = **True**)

Ahm.set\_index('Date', inplace = **True**)

Ahm=Ahm.astype('float64')

learning\_rate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

max\_depth = [int(x) **for** x **in** np.linspace(5, 30, num=6)]

subsample = [0.7, 0.6, 0.8]

min\_child\_weight = list(range(3, 8))

objective = ['reg:squarederror']

params = {

'n\_estimators': n\_estimators,

'learning\_rate': learning\_rate,

'max\_depth': max\_depth,

Ahm=Ahm.resample(rule='MS').mean()

ax=Ahm[['AQI\_calculated']].plot(figsize=(16,12),grid=**True**,lw=2,color='Red')

ax.autoscale(enable=**True**, axis='both', tight=**True**)

X = Ahm.iloc[:, :-1]

y = Ahm.iloc[:, -1]

'subsample': subsample,

'min\_child\_weight': min\_child\_weight,

'objective': objective

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.3,

random\_state=43)

}

xgb = XGBRegressor()

xgb.fit(X\_train, y\_train)

search = RandomizedSearchCV(xgb, params,

scoring='neg\_mean\_squared\_error',

cv=5, n\_iter=100, random\_state=43, n\_jobs=-1,

verbose=**True**)

f'Coefficient of determination R^2 on train set **{**xgb.score(X\_train,

y\_train)**}**'

f'Coefficient of determination R^2 on test set **{**xgb.score(X\_test, y\_test)**}**'

score = cross\_val\_score(xgb, X, y, cv = 3)

score.mean()

pred = xgb.predict(X\_test)

search.fit(X,y)

search.best\_params\_

search.best\_score\_

pred = search.predict(X\_test)

sns.distplot(y\_test-pred)

sns.distplot(y\_test - pred)

**Final Result**

pred = search.predict(X\_test)

Mean Abs Error: 0.0033662200716981887

Mean Sq Error: 0.00011384331947930463

Root Mean Error: 0.010669738491608153

print(f"Mean Abs Error: **{**metrics.mean\_absolute\_error(y\_test, pred)**}**")

print(f"Mean Sq Error: **{**metrics.mean\_squared\_error(y\_test, pred)**}**")

print(f"Root Mean Error: **{**np.sqrt(metrics.mean\_squared\_error(y\_test,

pred))**}**")





**Citywise mean squared error**

**(ii) Stacked LSTM**

LSTMs are widely used for sequence prediction problem. The stacked LSTM model was capable of

forecasting future days AQI for different cities on basis of past AQI information available.

**Citywise Mean Squared error**





**Thank You !**

[Github link for our project]

<https://github.com/Haaabs/FTS-Air-Quality-Index-Prediction>

[Drive link for our project]

<https://drive.google.com/drive/folders/1F2tTiHf2wsl7PRcYZBMs1Qb6jrForROg>

[References]

[https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYB](https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYBRID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI)

[RID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI](https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYBRID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI)

