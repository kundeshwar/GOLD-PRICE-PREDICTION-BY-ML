#-------------------------------------ABOUT DATASET

#Data Overview: This data file is a Comma separated value file format with 2290 rows and 7 columns. 
# It contains 5 columns which are numerical in datatype and one column in Date format. 
# Clearly the data shows value of the variables SPX,GLD,USO,SLV,EUR/USD against the dates in the date column.

#--------------------------------------work flow

#data collecting processes
#data anlysis
#data pre-processing
#data separation
#train-test-split
#model implimenation
#prediction of train-test data

#------------------------------------------import labrary
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
#--------------------------------------collection of data
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/gld_price_data.csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())
print(data.head(5))
print(data.tail(5))
print(data.isnull().sum())
#correlation between data
#positive and negative correlation
correlation = data.corr()
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size':8}, cmap="Blues")#fmt means how many decimal values we want
plt.show()
print(correlation["GLD"])
#checking distribution of Gld
sns.displot(data["GLD"], color="green")
plt.show()
#---------------------------------------separtion
x = data.drop(columns=["Date", "GLD"], axis=1)
print(x.head(5))
y = data["GLD"]
#-------------------------------------train-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
print(x.shape, x_test.shape, x_train.shape)

#------------------------------------model use
model = RandomForestRegressor(n_estimators=100)
model.fit(x_train, y_train)

#-----------------------------------prediction of data
y_train_p = model.predict(x_train)
print(y_train_p, "this is our train data", np.array(y_train), "this is original data")
accur = metrics.r2_score(y_train_p, y_train)#this is used to measure error score
print(accur)

#-----------------------------prediction of test data

y_test_p = model.predict(x_test)
accur = metrics.r2_score(y_test_p, y_test)
print(accur)

#-------------------------------------prediction of our data
x = [2107.780029, 18.490000, 15.580000,1.117905]
x_new= np.asarray(x)
x_reshape = x_new.reshape(1,-1)
y_pred = model.predict(x_reshape)
print(y_pred)



