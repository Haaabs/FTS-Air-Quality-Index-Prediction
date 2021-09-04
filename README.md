<div style="position:absolute;left:50%;margin-left:-360px;top:0px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background01.jpg)</div>

<div style="position:absolute;left:150.86px;top:110.26px" class="cls_002"><span class="cls_002">Air Quality Index Prediction</span></div>

<div style="position:absolute;left:170.18px;top:151.34px" class="cls_003"><span class="cls_003">Machine Learning Internship at FTS</span></div>

<div style="position:absolute;left:492.43px;top:223.90px" class="cls_004"><span class="cls_004">By,</span></div>

<div style="position:absolute;left:510.43px;top:243.34px" class="cls_004"><span class="cls_004">Akshata Kotti</span></div>

<div style="position:absolute;left:510.43px;top:262.78px" class="cls_004"><span class="cls_004">Shubham Urmaliya</span></div>

<div style="position:absolute;left:510.43px;top:282.22px" class="cls_004"><span class="cls_004">Abhishek</span></div>

<div style="position:absolute;left:510.43px;top:301.63px" class="cls_005"><span class="cls_005">Shreya Basu</span></div>

<div style="position:absolute;left:510.43px;top:321.12px" class="cls_004"><span class="cls_004">Neha Kumari</span></div>

<div style="position:absolute;left:510.43px;top:340.56px" class="cls_004"><span class="cls_004">Lokesh</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:415px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background02.jpg)</div>

<div style="position:absolute;left:49.39px;top:23.14px" class="cls_006"><span class="cls_006">Contents</span></div>

<div style="position:absolute;left:61.61px;top:80.38px" class="cls_007"><span class="cls_007">● Problem Statement</span></div>

<div style="position:absolute;left:61.61px;top:109.18px" class="cls_007"><span class="cls_007">● Data Visualisation before preprocessing</span></div>

<div style="position:absolute;left:61.61px;top:137.98px" class="cls_007"><span class="cls_007">● Data Visualisation on AQI</span></div>

<div style="position:absolute;left:61.61px;top:166.80px" class="cls_007"><span class="cls_007">● Data preprocessing</span></div>

<div style="position:absolute;left:97.61px;top:195.60px" class="cls_008"><span class="cls_008">❖</span> <span class="cls_007">Missing value treatment</span></div>

<div style="position:absolute;left:97.61px;top:224.38px" class="cls_009"><span class="cls_009">❖</span> <span class="cls_010">Air Quality Index(AQI) calculation</span></div>

<div style="position:absolute;left:97.61px;top:253.22px" class="cls_008"><span class="cls_008">❖</span> <span class="cls_007">Outlier treatment</span></div>

<div style="position:absolute;left:61.61px;top:282.02px" class="cls_007"><span class="cls_007">● Data Visualisation after preprocessing</span></div>

<div style="position:absolute;left:61.61px;top:310.80px" class="cls_010"><span class="cls_010">● Model Making</span></div>

<div style="position:absolute;left:97.61px;top:339.65px" class="cls_008"><span class="cls_008">❖</span> <span class="cls_007">XGBoost</span></div>

<div style="position:absolute;left:97.61px;top:368.45px" class="cls_008"><span class="cls_008">❖</span> <span class="cls_007">Stacked LSTM</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:830px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background03.jpg)</div>

<div style="position:absolute;left:55.37px;top:22.46px" class="cls_011"><span class="cls_011">Problem Statement</span></div>

<div style="position:absolute;left:46.01px;top:80.88px" class="cls_012"><span class="cls_012">To create a model which will predict the Air Quality Index (AQI).</span></div>

<div style="position:absolute;left:55.37px;top:126.58px" class="cls_007"><span class="cls_007">We were given two datasets:</span></div>

<div style="position:absolute;left:65.33px;top:145.78px" class="cls_007"><span class="cls_007">1\. cities_by_day → day-wise information including the amount of various chemical substances</span></div>

<div style="position:absolute;left:91.37px;top:164.96px" class="cls_010"><span class="cls_010">present in different cities and the AQI information.</span></div>

<div style="position:absolute;left:65.33px;top:184.20px" class="cls_007"><span class="cls_007">2\. cities_by_hours</span></div>

<div style="position:absolute;left:202.39px;top:184.20px" class="cls_007"><span class="cls_007">→ hours-wise information including the amount of various chemical</span></div>

<div style="position:absolute;left:91.37px;top:203.40px" class="cls_007"><span class="cls_007">substances present in different cities and the AQI information.</span></div>

<div style="position:absolute;left:55.37px;top:241.80px" class="cls_007"><span class="cls_007">We have initially performed Exploratory Data Analysis including Data preprocessing, Outlier</span></div>

<div style="position:absolute;left:55.37px;top:260.98px" class="cls_010"><span class="cls_010">treatment and Data visualization to study the datasets.</span></div>

<div style="position:absolute;left:55.37px;top:280.22px" class="cls_007"><span class="cls_007">We have then used certain algorithms like XGBoost and Stacked LSTM to create a model that</span></div>

<div style="position:absolute;left:55.37px;top:299.42px" class="cls_007"><span class="cls_007">will predict the AQI for any future reference using the input we are giving.</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:1245px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background04.jpg)</div>

<div style="position:absolute;left:31.68px;top:36.56px" class="cls_013"><span class="cls_013">Data Visualisation before preprocessing</span></div>

<div style="position:absolute;left:36.48px;top:83.23px" class="cls_014"><span class="cls_014">We used some visualisation techniques to understand the trends and relationships between different</span></div>

<div style="position:absolute;left:36.48px;top:100.03px" class="cls_014"><span class="cls_014">columns. The results are following.</span></div>

<div style="position:absolute;left:47.52px;top:116.83px" class="cls_014"><span class="cls_014">● There are a lot of missing values for xylene,PM2.5 and NH3\. But after looking at correlations</span></div>

<div style="position:absolute;left:72.48px;top:133.61px" class="cls_015"><span class="cls_015">AQI is reasonably dependent on these gases. So it is not good to drop these columns.</span></div>

<div style="position:absolute;left:47.52px;top:150.46px" class="cls_014"><span class="cls_014">● The second image is a plot of PM(PM2.5 +PM10) with months. From this graph we can see that</span></div>

<div style="position:absolute;left:72.48px;top:167.26px" class="cls_014"><span class="cls_014">values are not missing at random they are missing for long periods of time from this we found</span></div>

<div style="position:absolute;left:72.48px;top:184.06px" class="cls_014"><span class="cls_014">that the imputation methods like linear interpolation will not give realistic results and we started</span></div>

<div style="position:absolute;left:72.48px;top:200.86px" class="cls_014"><span class="cls_014">thinking about methods like KNN imputation.</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:1660px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background05.jpg)</div>

<div style="position:absolute;left:39.82px;top:26.43px" class="cls_013"><span class="cls_013">Data Visualisation on AQI</span></div>

<div style="position:absolute;left:26.33px;top:54.00px" class="cls_016"><span class="cls_016"></span>[Data visualization is the graphical representation of information and data. We use different visual elements like charts,](https://www.tableau.com/learn/articles/data-visualization/glossary)</div>

<div style="position:absolute;left:26.33px;top:70.80px" class="cls_016"><span class="cls_016"></span>[graphs, and maps, data visualization tools to provide an accessible way to see and understand trends, outliers, and](https://www.tableau.com/learn/articles/data-visualization/glossary)</div>

<div style="position:absolute;left:26.33px;top:87.60px" class="cls_016"><span class="cls_016">patterns in data.</span></div>

<div style="position:absolute;left:26.33px;top:104.40px" class="cls_016"><span class="cls_016">Visualization has been done on the dataset of cities_by_day to study certain trends. Some screenshots have been</span></div>

<div style="position:absolute;left:26.33px;top:121.18px" class="cls_017"><span class="cls_017">attached herewith. The link to the file has been given here:</span></div>

<div style="position:absolute;left:26.33px;top:138.02px" class="cls_064"><span class="cls_064"></span>[https://colab.research.google.com/drive/1UIySiXXD82j0ocehY9wtBLlZj7am7gl-#scrollTo=RHBP32Q3qcLu](https://colab.research.google.com/drive/1UIySiXXD82j0ocehY9wtBLlZj7am7gl-#scrollTo=RHBP32Q3qcLu)</div>

<div style="position:absolute;left:28.30px;top:370.61px" class="cls_019"><span class="cls_019">Calculating the proportion</span></div>

<div style="position:absolute;left:186.00px;top:370.61px" class="cls_019"><span class="cls_019">Grouping the cities based on average AQI</span></div>

<div style="position:absolute;left:484.13px;top:370.61px" class="cls_019"><span class="cls_019">Pie-chart showing distribution of</span></div>

<div style="position:absolute;left:28.30px;top:385.01px" class="cls_019"><span class="cls_019">of missing values</span></div>

<div style="position:absolute;left:484.13px;top:385.01px" class="cls_019"><span class="cls_019">pollutant in top polluted cities</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:2075px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background06.jpg)</div>

<div style="position:absolute;left:34.20px;top:31.49px" class="cls_013"><span class="cls_013">Data preprocessing</span></div>

<div style="position:absolute;left:44.14px;top:93.96px" class="cls_020"><span class="cls_020">KNN Imputation</span></div>

<div style="position:absolute;left:368.18px;top:96.89px" class="cls_020"><span class="cls_020">Outlier Detection Using Quantile Regression</span></div>

<div style="position:absolute;left:43.32px;top:121.10px" class="cls_021"><span class="cls_021">def</span></div>

<div style="position:absolute;left:61.92px;top:121.10px" class="cls_022"><span class="cls_022">fun</span><span class="cls_023">(dframe):</span></div>

<div style="position:absolute;left:368.47px;top:129.31px" class="cls_023"><span class="cls_023">Q1</span><span class="cls_024">=</span><span class="cls_023">df[</span><span class="cls_027">'AQI_calculated'</span><span class="cls_023">]</span><span class="cls_024">.</span><span class="cls_023">quantile(</span><span class="cls_024">0.25</span><span class="cls_023">)</span></div>

<div style="position:absolute;left:49.08px;top:133.70px" class="cls_023"><span class="cls_023">lis</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:70.92px;top:133.70px" class="cls_023"><span class="cls_023">[]</span></div>

<div style="position:absolute;left:368.47px;top:141.91px" class="cls_023"><span class="cls_023">Q3</span><span class="cls_024">=</span><span class="cls_023">df[</span><span class="cls_027">'AQI_calculated'</span><span class="cls_023">]</span><span class="cls_024">.</span><span class="cls_023">quantile(</span><span class="cls_024">0.75</span><span class="cls_023">)</span></div>

<div style="position:absolute;left:49.08px;top:146.30px" class="cls_021"><span class="cls_021">for</span> <span class="cls_023">i</span> <span class="cls_025">in</span></div>

<div style="position:absolute;left:83.28px;top:146.30px" class="cls_026"><span class="cls_026">range</span><span class="cls_023">(</span><span class="cls_024">0</span><span class="cls_023">, dframe</span><span class="cls_024">.</span><span class="cls_023">shape[</span><span class="cls_024">1</span><span class="cls_023">]):</span></div>

<div style="position:absolute;left:368.47px;top:154.51px" class="cls_023"><span class="cls_023">IQR</span><span class="cls_024">=</span><span class="cls_023">Q3</span><span class="cls_024">-</span><span class="cls_023">Q1</span></div>

<div style="position:absolute;left:60.60px;top:158.90px" class="cls_021"><span class="cls_021">if</span><span class="cls_023">(dframe</span><span class="cls_024">.</span><span class="cls_023">iloc[:,i]</span><span class="cls_024">.</span><span class="cls_023">dtypes</span> <span class="cls_024">==</span></div>

<div style="position:absolute;left:187.70px;top:158.90px" class="cls_027"><span class="cls_027">'object'</span><span class="cls_023">):</span></div>

<div style="position:absolute;left:368.47px;top:167.11px" class="cls_026"><span class="cls_026">print</span><span class="cls_023">(Q1)</span></div>

<div style="position:absolute;left:66.36px;top:171.50px" class="cls_023"><span class="cls_023">dframe</span><span class="cls_024">.</span><span class="cls_023">iloc[:,i]</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:143.52px;top:171.50px" class="cls_023"><span class="cls_023">pd</span><span class="cls_024">.</span><span class="cls_023">Categorical(dframe</span><span class="cls_024">.</span><span class="cls_023">iloc[:,i])</span></div>

<div style="position:absolute;left:368.47px;top:179.71px" class="cls_026"><span class="cls_026">print</span><span class="cls_023">(Q3)</span></div>

<div style="position:absolute;left:66.36px;top:184.10px" class="cls_023"><span class="cls_023">dframe</span><span class="cls_024">.</span><span class="cls_023">iloc[:,i]</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:143.52px;top:184.10px" class="cls_023"><span class="cls_023">dframe</span><span class="cls_024">.</span><span class="cls_023">iloc[:,i]</span><span class="cls_024">.</span><span class="cls_023">cat</span><span class="cls_024">.</span><span class="cls_023">codes</span></div>

<div style="position:absolute;left:368.47px;top:192.31px" class="cls_026"><span class="cls_026">print</span><span class="cls_023">(IQR)</span></div>

<div style="position:absolute;left:66.36px;top:196.68px" class="cls_028"><span class="cls_028">dframe</span><span class="cls_029">.</span><span class="cls_028">iloc[:,i]</span> <span class="cls_029">=</span></div>

<div style="position:absolute;left:143.52px;top:196.68px" class="cls_028"><span class="cls_028">dframe</span><span class="cls_029">.</span><span class="cls_028">iloc[:,i]</span><span class="cls_029">.</span><span class="cls_028">astype(</span><span class="cls_030">'object'</span><span class="cls_028">)</span></div>

<div style="position:absolute;left:368.47px;top:204.89px" class="cls_028"><span class="cls_028">Lower_Whisker</span> <span class="cls_029">=</span></div>

<div style="position:absolute;left:452.98px;top:204.89px" class="cls_028"><span class="cls_028">Q1</span> <span class="cls_029">-</span></div>

<div style="position:absolute;left:476.26px;top:204.89px" class="cls_029"><span class="cls_029">1.5*</span><span class="cls_028">IQR</span></div>

<div style="position:absolute;left:368.47px;top:217.54px" class="cls_023"><span class="cls_023">Upper_Whisker</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:452.98px;top:217.54px" class="cls_023"><span class="cls_023">Q3</span> <span class="cls_024">+</span></div>

<div style="position:absolute;left:478.90px;top:217.54px" class="cls_024"><span class="cls_024">1.5*</span><span class="cls_023">IQR</span></div>

<div style="position:absolute;left:66.36px;top:221.93px" class="cls_023"><span class="cls_023">lis</span><span class="cls_024">.</span><span class="cls_023">append(dframe</span><span class="cls_024">.</span><span class="cls_023">columns[i])</span></div>

<div style="position:absolute;left:368.47px;top:230.14px" class="cls_026"><span class="cls_026">print</span><span class="cls_023">(Lower_Whisker, Upper_Whisker)</span></div>

<div style="position:absolute;left:49.08px;top:234.53px" class="cls_023"><span class="cls_023">KNN</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:83.16px;top:234.53px" class="cls_023"><span class="cls_023">KNNImputer(n_neighbors</span><span class="cls_024">=3</span><span class="cls_023">)</span></div>

<div style="position:absolute;left:368.47px;top:243.46px" class="cls_023"><span class="cls_023">df</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:389.14px;top:243.46px" class="cls_023"><span class="cls_023">df[df[</span><span class="cls_027">'AQI_calculated'</span><span class="cls_023">]</span><span class="cls_024"><</span></div>

<div style="position:absolute;left:499.66px;top:243.46px" class="cls_023"><span class="cls_023">Upper_Whisker]</span></div>

<div style="position:absolute;left:49.08px;top:247.13px" class="cls_023"><span class="cls_023">dframe</span> <span class="cls_024">=</span></div>

<div style="position:absolute;left:93.72px;top:247.13px" class="cls_023"><span class="cls_023">pd</span><span class="cls_024">.</span><span class="cls_023">DataFrame(KNN</span><span class="cls_024">.</span><span class="cls_023">fit_transform(dframe))</span></div>

<div style="position:absolute;left:49.08px;top:260.45px" class="cls_021"><span class="cls_021">return</span></div>

<div style="position:absolute;left:82.32px;top:260.45px" class="cls_023"><span class="cls_023">dframe</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:2490px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background07.jpg)</div>

<div style="position:absolute;left:49.92px;top:44.59px" class="cls_031"><span class="cls_031">Data Preprocessing of Cities_by_day and Cities_by_hours dataset</span></div>

<div style="position:absolute;left:49.92px;top:80.38px" class="cls_032"><span class="cls_032">1] Missing value treatment</span><span class="cls_033">:</span> <span class="cls_034">Methods used to treat missing values are:</span></div>

<div style="position:absolute;left:60.96px;top:107.02px" class="cls_034"><span class="cls_034">● Citywise Mean imputation</span></div>

<div style="position:absolute;left:60.96px;top:132.22px" class="cls_034"><span class="cls_034">● Citywise Linear interpolation</span></div>

<div style="position:absolute;left:60.96px;top:157.40px" class="cls_035"><span class="cls_035">● Citywise K-Nearest Neighbors(KNN) imputation</span></div>

<div style="position:absolute;left:53.40px;top:183.00px" class="cls_032"><span class="cls_032">2] AQI calculation:</span> <span class="cls_034">AQI is the maximum of sub-indices calculated for individual pollutants.</span></div>

<div style="position:absolute;left:53.40px;top:210.00px" class="cls_032"><span class="cls_032">3] Outlier treatment</span><span class="cls_034">: Outliers were detected and treated using Quantile Regression.</span></div>

<div style="position:absolute;left:49.92px;top:235.92px" class="cls_019"><span class="cls_019">Percentage of missing</span></div>

<div style="position:absolute;left:317.34px;top:235.92px" class="cls_019"><span class="cls_019">Percentage of missing</span></div>

<div style="position:absolute;left:49.92px;top:257.54px" class="cls_019"><span class="cls_019">values in cities_by_day:</span></div>

<div style="position:absolute;left:315.19px;top:257.54px" class="cls_019"><span class="cls_019">values in cities_by_hour:</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:2905px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background08.jpg)</div>

<div style="position:absolute;left:29.98px;top:33.22px" class="cls_006"><span class="cls_006">Data Visualisation after preprocessing</span></div>

<div style="position:absolute;left:33.36px;top:73.61px" class="cls_036"><span class="cls_036">Visualization has also been performed after preprocessing the dataset cities_by_hours i.e., removing the</span></div>

<div style="position:absolute;left:33.36px;top:90.46px" class="cls_037"><span class="cls_037">missing values in the dataset.</span></div>

<div style="position:absolute;left:15.62px;top:346.13px" class="cls_019"><span class="cls_019">Proportion of missing</span></div>

<div style="position:absolute;left:508.66px;top:346.87px" class="cls_019"><span class="cls_019">Pie-chart showing imputed AQI</span></div>

<div style="position:absolute;left:240.02px;top:356.26px" class="cls_019"><span class="cls_019">Correlation analysis</span></div>

<div style="position:absolute;left:15.62px;top:360.53px" class="cls_019"><span class="cls_019">values has been reduced</span></div>

<div style="position:absolute;left:517.66px;top:361.27px" class="cls_019"><span class="cls_019">values for top polluted cities</span></div>

<div style="position:absolute;left:15.62px;top:374.93px" class="cls_019"><span class="cls_019">to zero</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:3320px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background09.jpg)</div>

<div style="position:absolute;left:29.14px;top:20.57px" class="cls_013"><span class="cls_013">Model Making - (i) XGBoost Regressor</span></div>

<div style="position:absolute;left:33.55px;top:70.99px" class="cls_038"><span class="cls_038">def</span></div>

<div style="position:absolute;left:51.31px;top:70.99px" class="cls_039"><span class="cls_039">fun</span><span class="cls_040">(Ahm):</span></div>

<div style="position:absolute;left:404.38px;top:69.89px" class="cls_043"><span class="cls_043">n_estimators</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:462.94px;top:69.89px" class="cls_043"><span class="cls_043">[</span><span class="cls_047">int</span><span class="cls_043">(x)</span> <span class="cls_048">for</span></div>

<div style="position:absolute;left:500.50px;top:69.89px" class="cls_043"><span class="cls_043">x</span> <span class="cls_049">in</span></div>

<div style="position:absolute;left:517.08px;top:69.89px" class="cls_043"><span class="cls_043">np</span><span class="cls_044">.</span><span class="cls_043">linspace(start</span><span class="cls_044">=100</span><span class="cls_043">, stop</span><span class="cls_044">=1200</span><span class="cls_043">, num</span><span class="cls_044">=12</span><span class="cls_043">)]</span></div>

<div style="position:absolute;left:39.07px;top:80.59px" class="cls_040"><span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">drop([</span><span class="cls_042">'City'</span><span class="cls_040">],axis</span><span class="cls_041">=1</span><span class="cls_040">,inplace</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:189.58px;top:80.59px" class="cls_038"><span class="cls_038">True</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:404.38px;top:80.07px" class="cls_050"><span class="cls_050">learning_rate</span></div>

<div style="position:absolute;left:456.58px;top:80.07px" class="cls_051"><span class="cls_051">=</span></div>

<div style="position:absolute;left:463.90px;top:80.07px" class="cls_050"><span class="cls_050">[</span><span class="cls_051">0.05</span><span class="cls_050">,</span> <span class="cls_051">0.1</span><span class="cls_050">,</span> <span class="cls_051">0.2</span><span class="cls_050">,</span> <span class="cls_051">0.3</span><span class="cls_050">,</span> <span class="cls_051">0.4</span><span class="cls_050">,</span> <span class="cls_051">0.5</span><span class="cls_050">,</span> <span class="cls_051">0.6</span><span class="cls_050">]</span></div>

<div style="position:absolute;left:39.07px;top:90.19px" class="cls_040"><span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">set_index(</span><span class="cls_042">'Date'</span><span class="cls_040">, inplace</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:181.30px;top:90.19px" class="cls_038"><span class="cls_038">True</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:404.38px;top:90.31px" class="cls_043"><span class="cls_043">max_depth</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:456.34px;top:90.31px" class="cls_043"><span class="cls_043">[</span><span class="cls_047">int</span><span class="cls_043">(x)</span> <span class="cls_048">for</span></div>

<div style="position:absolute;left:493.78px;top:90.31px" class="cls_043"><span class="cls_043">x</span> <span class="cls_049">in</span></div>

<div style="position:absolute;left:510.48px;top:90.31px" class="cls_043"><span class="cls_043">np</span><span class="cls_044">.</span><span class="cls_043">linspace(</span><span class="cls_044">5</span><span class="cls_043">,</span> <span class="cls_044">30</span><span class="cls_043">, num</span><span class="cls_044">=6</span><span class="cls_043">)]</span></div>

<div style="position:absolute;left:39.07px;top:99.79px" class="cls_040"><span class="cls_040">Ahm</span><span class="cls_041">=</span><span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">astype(</span><span class="cls_042">'float64'</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:404.38px;top:100.51px" class="cls_043"><span class="cls_043">subsample</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:455.38px;top:100.51px" class="cls_043"><span class="cls_043">[</span><span class="cls_044">0.7</span><span class="cls_043">,</span> <span class="cls_044">0.6</span><span class="cls_043">,</span> <span class="cls_044">0.8</span><span class="cls_043">]</span></div>

<div style="position:absolute;left:39.07px;top:109.39px" class="cls_040"><span class="cls_040">Ahm</span><span class="cls_041">=</span><span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">resample(rule</span><span class="cls_041">=</span><span class="cls_042">'MS'</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:177.70px;top:109.39px" class="cls_041"><span class="cls_041">.</span><span class="cls_040">mean()</span></div>

<div style="position:absolute;left:404.38px;top:110.71px" class="cls_043"><span class="cls_043">min_child_weight</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:479.50px;top:110.71px" class="cls_047"><span class="cls_047">list</span><span class="cls_043">(</span><span class="cls_047">range</span><span class="cls_043">(</span><span class="cls_044">3</span><span class="cls_043">,</span> <span class="cls_044">8</span><span class="cls_043">))</span></div>

<div style="position:absolute;left:404.38px;top:120.91px" class="cls_043"><span class="cls_043">objective</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:447.70px;top:120.91px" class="cls_043"><span class="cls_043">[</span><span class="cls_045">'reg:squarederror'</span><span class="cls_043">]</span></div>

<div style="position:absolute;left:33.55px;top:128.62px" class="cls_040"><span class="cls_040">ax</span><span class="cls_041">=</span><span class="cls_040">Ahm[[</span><span class="cls_042">'AQI_calculated'</span><span class="cls_040">]]</span><span class="cls_041">.</span><span class="cls_040">plot(figsize</span><span class="cls_041">=</span><span class="cls_040">(</span><span class="cls_041">16</span><span class="cls_040">,</span><span class="cls_041">12</span><span class="cls_040">),grid</span><span class="cls_041">=</span><span class="cls_038">True</span><span class="cls_040">,lw</span><span class="cls_041">=2</span><span class="cls_040">,color</span><span class="cls_041">=</span><span class="cls_042">'Red'</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:404.38px;top:131.11px" class="cls_043"><span class="cls_043">params</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:442.54px;top:131.11px" class="cls_043"><span class="cls_043">{</span></div>

<div style="position:absolute;left:39.07px;top:138.22px" class="cls_040"><span class="cls_040">ax</span><span class="cls_041">.</span><span class="cls_040">autoscale(enable</span><span class="cls_041">=</span><span class="cls_038">True</span><span class="cls_040">, axis</span><span class="cls_041">=</span><span class="cls_042">'both'</span><span class="cls_040">, tight</span><span class="cls_041">=</span><span class="cls_038">True</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:413.86px;top:141.31px" class="cls_045"><span class="cls_045">'n_estimators'</span><span class="cls_043">: n_estimators,</span></div>

<div style="position:absolute;left:39.07px;top:147.82px" class="cls_040"><span class="cls_040">X</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:57.07px;top:147.82px" class="cls_040"><span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">iloc[:, :</span><span class="cls_041">-1</span><span class="cls_040">]</span></div>

<div style="position:absolute;left:413.86px;top:151.51px" class="cls_045"><span class="cls_045">'learning_rate'</span><span class="cls_043">: learning_rate,</span></div>

<div style="position:absolute;left:39.07px;top:157.42px" class="cls_040"><span class="cls_040">y</span> <span class="cls_041">=</span> <span class="cls_040">Ahm</span><span class="cls_041">.</span><span class="cls_040">iloc[:,</span> <span class="cls_041">-1</span><span class="cls_040">]</span></div>

<div style="position:absolute;left:413.86px;top:161.71px" class="cls_045"><span class="cls_045">'max_depth'</span><span class="cls_043">: max_depth,</span></div>

<div style="position:absolute;left:413.86px;top:171.89px" class="cls_052"><span class="cls_052">'subsample'</span><span class="cls_050">: subsample,</span></div>

<div style="position:absolute;left:39.07px;top:176.62px" class="cls_040"><span class="cls_040">X_train, X_test, y_train, y_test</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:182.98px;top:176.62px" class="cls_040"><span class="cls_040">train_test_split(X, y, test_size</span><span class="cls_041">=0.3</span><span class="cls_040">,</span></div>

<div style="position:absolute;left:413.86px;top:182.14px" class="cls_045"><span class="cls_045">'min_child_weight'</span><span class="cls_043">: min_child_weight,</span></div>

<div style="position:absolute;left:33.55px;top:186.22px" class="cls_040"><span class="cls_040">random_state</span><span class="cls_041">=43</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:413.86px;top:192.34px" class="cls_045"><span class="cls_045">'objective'</span><span class="cls_043">: objective</span></div>

<div style="position:absolute;left:404.38px;top:202.54px" class="cls_043"><span class="cls_043">}</span></div>

<div style="position:absolute;left:39.07px;top:205.44px" class="cls_040"><span class="cls_040">xgb</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:66.43px;top:205.44px" class="cls_040"><span class="cls_040">XGBRegressor()</span></div>

<div style="position:absolute;left:39.07px;top:215.04px" class="cls_040"><span class="cls_040">xgb</span><span class="cls_041">.</span><span class="cls_040">fit(X_train, y_train)</span></div>

<div style="position:absolute;left:404.38px;top:222.94px" class="cls_043"><span class="cls_043">search</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:439.66px;top:222.94px" class="cls_043"><span class="cls_043">RandomizedSearchCV(xgb, params,</span></div>

<div style="position:absolute;left:39.07px;top:224.64px" class="cls_040"><span class="cls_040">f</span><span class="cls_042">'Coefficient of determination R^2 on train set</span> <span class="cls_046">{</span><span class="cls_040">xgb</span><span class="cls_041">.</span><span class="cls_040">score(X_train, y_train)</span><span class="cls_046">}</span><span class="cls_042">'</span></div>

<div style="position:absolute;left:399.58px;top:233.14px" class="cls_043"><span class="cls_043">scoring</span><span class="cls_044">=</span><span class="cls_045">'neg_mean_squared_error'</span><span class="cls_043">,</span></div>

<div style="position:absolute;left:39.07px;top:234.24px" class="cls_040"><span class="cls_040">f</span><span class="cls_042">'Coefficient of determination R^2 on test set</span> <span class="cls_046">{</span><span class="cls_040">xgb</span><span class="cls_041">.</span><span class="cls_040">score(X_test, y_test)</span><span class="cls_046">}</span><span class="cls_042">'</span></div>

<div style="position:absolute;left:468.46px;top:243.34px" class="cls_043"><span class="cls_043">cv</span><span class="cls_044">=5</span><span class="cls_043">, n_iter</span><span class="cls_044">=100</span><span class="cls_043">, random_state</span><span class="cls_044">=43</span><span class="cls_043">, n_jobs</span><span class="cls_044">=-1</span><span class="cls_043">,</span></div>

<div style="position:absolute;left:39.07px;top:253.44px" class="cls_040"><span class="cls_040">score</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:74.83px;top:253.44px" class="cls_040"><span class="cls_040">cross_val_score(xgb, X, y, cv</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:215.74px;top:253.44px" class="cls_041"><span class="cls_041">3</span><span class="cls_040">)</span></div>

<div style="position:absolute;left:399.58px;top:253.52px" class="cls_050"><span class="cls_050">verbose</span><span class="cls_051">=</span><span class="cls_053">True</span><span class="cls_050">)</span></div>

<div style="position:absolute;left:39.07px;top:263.04px" class="cls_040"><span class="cls_040">score</span><span class="cls_041">.</span><span class="cls_040">mean()</span></div>

<div style="position:absolute;left:404.38px;top:263.76px" class="cls_043"><span class="cls_043">search</span><span class="cls_044">.</span><span class="cls_043">fit(X,y)</span></div>

<div style="position:absolute;left:39.07px;top:272.64px" class="cls_040"><span class="cls_040">pred</span> <span class="cls_041">=</span></div>

<div style="position:absolute;left:70.27px;top:272.64px" class="cls_040"><span class="cls_040">xgb</span><span class="cls_041">.</span><span class="cls_040">predict(X_test)</span></div>

<div style="position:absolute;left:404.38px;top:273.96px" class="cls_043"><span class="cls_043">search</span><span class="cls_044">.</span><span class="cls_043">best_params_</span></div>

<div style="position:absolute;left:404.38px;top:284.16px" class="cls_043"><span class="cls_043">search</span><span class="cls_044">.</span><span class="cls_043">best_score_</span></div>

<div style="position:absolute;left:39.07px;top:291.86px" class="cls_040"><span class="cls_040">sns</span><span class="cls_041">.</span><span class="cls_040">distplot(y_test</span> <span class="cls_041">-</span></div>

<div style="position:absolute;left:127.54px;top:291.86px" class="cls_040"><span class="cls_040">pred)</span></div>

<div style="position:absolute;left:404.38px;top:294.36px" class="cls_043"><span class="cls_043">pred</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:431.26px;top:294.36px" class="cls_043"><span class="cls_043">search</span><span class="cls_044">.</span><span class="cls_043">predict(X_test)</span></div>

<div style="position:absolute;left:404.38px;top:304.56px" class="cls_043"><span class="cls_043">sns</span><span class="cls_044">.</span><span class="cls_043">distplot(y_test</span></div>

<div style="position:absolute;left:472.66px;top:304.56px" class="cls_044"><span class="cls_044">-</span><span class="cls_043">pred)</span></div>

<div style="position:absolute;left:39.26px;top:303.58px" class="cls_054"><span class="cls_054">Final Result</span></div>

<div style="position:absolute;left:404.38px;top:324.96px" class="cls_043"><span class="cls_043">pred</span> <span class="cls_044">=</span></div>

<div style="position:absolute;left:431.26px;top:324.96px" class="cls_043"><span class="cls_043">search</span><span class="cls_044">.</span><span class="cls_043">predict(X_test)</span></div>

<div style="position:absolute;left:404.38px;top:335.14px" class="cls_055"><span class="cls_055">print</span><span class="cls_050">(f</span><span class="cls_052">"Mean Abs Error:</span> <span class="cls_056">{</span><span class="cls_050">metrics</span><span class="cls_051">.</span><span class="cls_050">mean_absolute_error(y_test, pred)</span><span class="cls_056">}</span><span class="cls_052">"</span><span class="cls_050">)</span></div>

<div style="position:absolute;left:35.35px;top:336.41px" class="cls_058"><span class="cls_058">Mean Abs Error: 0.0033662200716981887</span></div>

<div style="position:absolute;left:404.38px;top:345.38px" class="cls_047"><span class="cls_047">print</span><span class="cls_043">(f</span><span class="cls_045">"Mean Sq Error:</span> <span class="cls_057">{</span><span class="cls_043">metrics</span><span class="cls_044">.</span><span class="cls_043">mean_squared_error(y_test, pred)</span><span class="cls_057">}</span><span class="cls_045">"</span><span class="cls_043">)</span></div>

<div style="position:absolute;left:35.35px;top:349.01px" class="cls_058"><span class="cls_058">Mean Sq Error: 0.00011384331947930463</span></div>

<div style="position:absolute;left:404.38px;top:356.06px" class="cls_047"><span class="cls_047">print</span><span class="cls_043">(f</span><span class="cls_045">"Root Mean Error:</span> <span class="cls_057">{</span><span class="cls_043">np</span><span class="cls_044">.</span><span class="cls_043">sqrt(metrics</span><span class="cls_044">.</span><span class="cls_043">mean_squared_error(y_test, pred))</span><span class="cls_057">}</span><span class="cls_045">"</span><span class="cls_043">)</span></div>

<div style="position:absolute;left:35.35px;top:362.33px" class="cls_058"><span class="cls_058">Root Mean Error: 0.010669738491608153</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:3735px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background10.jpg)</div>

<div style="position:absolute;left:379.25px;top:6.96px" class="cls_059"><span class="cls_059">Citywise mean squared error</span></div>

<div style="position:absolute;left:24.91px;top:35.72px" class="cls_013"><span class="cls_013">(ii) Stacked LSTM</span></div>

<div style="position:absolute;left:26.98px;top:75.77px" class="cls_036"><span class="cls_036">LSTMs are widely used for sequence prediction problem. The stacked LSTM model was capable of</span></div>

<div style="position:absolute;left:26.98px;top:92.62px" class="cls_037"><span class="cls_037">forecasting future days AQI for different cities on basis of past AQI information available.</span></div>

<div style="position:absolute;left:26.98px;top:128.26px" class="cls_060"><span class="cls_060">Citywise Mean Squared error</span></div>

</div>

<div style="position:absolute;left:50%;margin-left:-360px;top:4150px;width:720px;height:405px;border-style:outset;overflow:hidden">

<div style="position:absolute;left:0px;top:0px">![](0a8cedd8-0d57-11ec-a980-0cc47a792c0a_id_0a8cedd8-0d57-11ec-a980-0cc47a792c0a_files/background11.jpg)</div>

<div style="position:absolute;left:64.54px;top:104.83px" class="cls_061"><span class="cls_061">Thank You !</span></div>

<div style="position:absolute;left:64.54px;top:162.79px" class="cls_062"><span class="cls_062">[Github link for our project]</span></div>

<div style="position:absolute;left:64.54px;top:187.75px" class="cls_065"><span class="cls_065"></span>[https://github.com/Haaabs/FTS-Air-Quality-Index-Prediction](https://github.com/Haaabs/FTS-Air-Quality-Index-Prediction)</div>

<div style="position:absolute;left:66.72px;top:219.53px" class="cls_062"><span class="cls_062">[Drive link for our project]</span></div>

<div style="position:absolute;left:67.68px;top:250.73px" class="cls_065"><span class="cls_065"></span>[https://drive.google.com/drive/folders/1F2tTiHf2wsl7PRcYZBMs1Qb6jrForROg](https://drive.google.com/drive/folders/1F2tTiHf2wsl7PRcYZBMs1Qb6jrForROg)</div>

<div style="position:absolute;left:67.68px;top:281.95px" class="cls_062"><span class="cls_062">[References]</span></div>

<div style="position:absolute;left:61.20px;top:313.15px" class="cls_065"><span class="cls_065"></span>[https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYB](https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYBRID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI)</div>

<div style="position:absolute;left:61.20px;top:328.75px" class="cls_065"><span class="cls_065"></span>[RID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI](https://www.researchgate.net/publication/341990700_AIR_QUALITY_INDEX_FORECASTING_USING_HYBRID_NEURAL_NETWORK_MODEL_WITH_LSTM_ON_AQI)</div>

</div>