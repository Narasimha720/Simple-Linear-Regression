# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:25:42 2023

@author: narasimha_talabhaktula
"""
#Load the data
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df.shape
df.info()
df.head()
df.dtypes
df["Salary"]

#Data Visualisation
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
sns.distplot(df["YearsExperience"])

#EDA
#histogram
df["Salary"].hist()
df["Salary"].skew()
df["Salary"].kurt()
df["Salary"].describe()

df["YearsExperience"].describe()
df["YearsExperience"].hist()
df["YearsExperience"].skew()
df["YearsExperience"].kurt()

# scatter plot
import matplotlib.pyplot as plt
plt.scatter(x=df['YearsExperience'], y=df['Salary'])
plt.scatter(x=df['YearsExperience'], y=df['Salary'],color ='red')
plt.show()

#correlation
df[["Salary","YearsExperience"]].corr()
df.corr()

#box plot
df.boxplot(column='YearsExperience',vert=False) #There are no ouliers present
import numpy as np
np
Q1 = np.percentile(df["YearsExperience"],25)
Q3 = np.percentile(df["YearsExperience"],75)
IQR = Q3 - Q1
UW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df["YearsExperience"]>UW
df[df["YearsExperience"]>UW]
len(df[df["YearsExperience"]>UW])

# split the variable as X and Y
X = df[['YearsExperience']]

X[['YearsExperiencedSquared']] = X[['YearsExperience']]**2

X[['SquareRootYearsExperience']] = np.sqrt(X[['YearsExperience']])

X[['LogYearsExperience']] = np.log(X[['YearsExperience']])

X[['InverseSquareRootYearsExperience']] = 1 / np.sqrt(X[['YearsExperience']])
X

Y = df["Salary"]
Y

#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=5)
X_train
X_test
Y_train
Y_test

# fit the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train) # Bo + B1x1
LR.intercept_ # Bo
LR.coef_   #B1

# calc y_pred
Y_pred = LR.predict(X)
Y_pred
Y
#=============================================
# metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error:", mse.round(3))
print("Root Mean squared Error:", np.sqrt(mse).round(3))  
#5592.044 , 5590.841 ,5554.112 ,5533.673 ,5513.149#

#R-squared (R2) value for your linear regression model
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3))  #R square: 0.963
#95.696, 95.698 , 95.754 ,95.785,  95.816 


# plt the scatter plot with y_pred
import matplotlib.pyplot as plt
plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =df["Salary"] )
plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =Y_pred ,color = 'red' )
plt.plot(1 / np.sqrt(X['YearsExperience']),Y_pred,color='Black')    #x value is automatically taken by python###
plt.show()


# here in the above scenario, we applied a transformation on x variable and that transformation is 
#X['InverseSquareRootYearsExperience'] = 1 / np.sqrt(X['YearsExperience'])
#so after taking this transformed X value along with our target variable provides less mean squared error along with good R2 score.

