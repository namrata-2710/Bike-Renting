import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn import neighbors
# %matplotlib inline
data = pd.read_csv("day.csv", sep = ',',encoding='unicode_escape')
print(data.shape)#

#Missing Data Analysis
missing_val = pd.DataFrame(data.isnull().sum())
# print(missing_val)#No missing data
cnames = ["temp", "atemp", "hum", "windspeed", "casual", "registered"]
for name in cnames:
    data[name]= pd.to_numeric(data[name],errors='coerce')


# #Outlier Analysis

# for names in cnames:
#    plt.hist(data[names])
#    plt.show()

for name in cnames:
    q75,q25 = np.percentile(data.loc[:,name],[75,25])
    iqr = q75-q25
    min=q25-(iqr*1.5)
    max = q75 + (iqr * 1.5)
    data=data.drop(data[data.loc[:,name]<min].index)
    data = data.drop(data[data.loc[:, name] > max].index)

missing_val_after_outlier = pd.DataFrame(data.isnull().sum())
# print(missing_val_after_outlier)#No missing value means no outlier

##Feature Selection
#Correlation analysis for numerical data
df_cor=data.loc[:,cnames]
ax = plt.subplots(figsize = (7,5))
corr = df_cor.corr()
sns.heatmap(corr)
# plt.show()
#temp and atemp are highly correlated
#removing atemp
data = data.drop('atemp',axis=1)
# print(data.shape)
#Dropping dteday as we already have columns that are derived from this column
data = data.drop('dteday', axis=1)
#Feature Scaling
#Checking for uniform distribution
fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].hist(data['casual'])#non uniform
axs[0].set_title('casual')

axs[1].hist(data['registered'])#uniform
axs[1].set_title('registered')

# plt.show()

#normalization for casual beacuse it is left skewed
data['casual'] = (data['casual']-data['casual'].min())/(data['casual'].max()-data['casual'].min())
# print(data['casual'].head(5))


#Standardization  for registered
data['registered']= (data['registered'] - data['registered'].mean())/data['registered'].std()
# print(data['registered'].head(5))

##Sampling

#Stratified sampling
#select categorical variable
cat_var = data['holiday']
#select subset
train,test= train_test_split(data,test_size=0.3,stratify=cat_var)

# print(train.shape)
# print(test.shape)
# print(test.head(10))

#Decision Tree
fit = sklearn.tree.DecisionTreeRegressor(max_depth=500).fit(train.iloc[:,0:13],train.iloc[:,13])
# summary(fit)
predict_DT = fit.predict(test.iloc[:,0:13])
# print(predict_DT)
# # #MAPE FUnction
def MAPE(test_data, pred):
     mape = np.mean(np.abs((test_data-pred)/test_data))
     return mape
# print(MAPE(test.iloc[:,13],predict_DT))
#error : 0.03%

#Random Forest
regr = RandomForestRegressor(n_estimators=500,max_depth=2,random_state=0)
regr.fit(train.iloc[:,0:13],train.iloc[:,13])
predict_RF = regr.predict(test.iloc[:,0:13])
print(MAPE(test.iloc[:,13],predict_RF))
#error 0.53%

#linear regression
model = sm.OLS(train.iloc[:,13],train.iloc[:,0:13]).fit()
predict_LR = model.predict(test.iloc[:,0:13])
print(MAPE(test.iloc[:,13],predict_LR))
#error 0.07%

#KNN
#minimum error rate is at k = 8

model = neighbors.KNeighborsRegressor(n_neighbors = 8 )
model.fit(train.iloc[:,0:13], train.iloc[:,13])  #fit the model
predict_KNN=model.predict(test.iloc[:,0:13]) #make prediction on test set
print(MAPE(test.iloc[:,13],predict_KNN))
#error = 0.22%

predict_RF = pd.DataFrame(predict_RF)
