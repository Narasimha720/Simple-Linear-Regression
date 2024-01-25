# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:35:39 2023

@author: narasimha_talabhaktula
"""
#Load the data
import numpy as np
import pandas as pd
df = pd.read_csv("delivery_time.csv")
df.head()
df.info()
df.describe()
df.dtypes
df.isnull().sum()  #there are no null values
df["Delivery Time"]
df["Sorting Time"]

#Data Visualization
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

#distribution plot (histogram) 
sns.distplot(df["Sorting Time"])

#histogram
df["Delivery Time"].describe()
df["Delivery Time"].hist()
df["Delivery Time"].skew()   #0.3523900822831107 (POSITIVELY SKEWED)
df["Delivery Time"].kurt()   #0.31795982942685397 (LEPTO KURTOSIS)

df["Sorting Time"].describe()
df["Sorting Time"].hist()
df["Sorting Time"].skew()   #0.047115474210530174  (POSITIVELY SKEWED)
df["Sorting Time"].kurt()   #-1.14845514534878   (PLATY KURTOSIS)

# scatter plot
import matplotlib.pyplot as plt
plt.scatter(x=df['Sorting Time'], y=df['Delivery Time'],color="red")
plt.show()
df.corr()

#correlation
df[["Delivery Time","Sorting Time"]].corr()
df.corr()

#Box plot
df.boxplot(column='Sorting Time',vert=False)
import numpy as np
Q1 = np.percentile(df["Sorting Time"],25)
Q3 = np.percentile(df["Sorting Time"],75)
IQR = Q3 - Q1
UW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df["Sorting Time"]>UW
df[df["Sorting Time"]>UW]
len(df[df["Sorting Time"]>UW])  #there are no outliers 

#=====================================================================================================
# split the variable as X and Y
X = df[["Sorting Time"]]
X

#Transformation on X variable
X[["Sorting Time"]] = X[["Sorting Time"]]**2
#X[["Sorting Time"]] = np.sqrt(X[["Sorting Time"]])
#X[["Sorting Time"]] = np.log(X[["Sorting Time"]])
#X[["Sorting Time"]] = 1/np.sqrt(X[["Sorting Time"]])

df.dropna(subset=['Sorting Time'], inplace=True)
Y = df['Delivery Time']
Y
#=====================================================================================
# Split the data into training and testing sets (You should use cross-validation for better evaluation)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state=42)
#==================================================================================================
# fit the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y) # Bo + B1x1
LR.intercept_ # Bo
LR.coef_   #B1

#Predicted
Y_pred = LR.predict(X)
Y_pred 

#Actual 
Y

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error:", mse.round(3))
print("Root Mean squared Error:", np.sqrt(mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3))

# to fix the accuracy scores of training and testing data#
#validation set approach#
training_error = []
test_error = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state = i)
    X_train.shape
    X_test.shape
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print(training_error)
print(test_error)
print("average training error :",np.mean( training_error).round(3))   
print("average testing error :",np.mean( test_error).round(3))
#=====================================================================================================
# metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error:", mse.round(3))
print("Root Mean squared Error:", np.sqrt(mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3)) #0.632 (Poor Model)

#=====================================================================================================
#====================================================================================================
# plt the scatter plot with y_pred
#plotting #import matplotlib.pyplot as plt
plt.scatter(x =1 / np.sqrt(X[['Sorting Time']]),y =df["Delivery Time"] )
plt.scatter(x = 1 / np.sqrt(X[['Sorting Time']]),y =Y_pred ,color = 'red' )
plt.plot(1 / np.sqrt(X[['Sorting Time']]),Y_pred,color='Black')#x value is automatically taken by python###
plt.show()




























