# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:24:02 2024

@author: lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
df = pd.read_csv("/Users/lenovo/ML/50_Startups.csv")
x = df.iloc[:,:-1]
y = df.iloc[:,4]

#Convert the data to categorical colum
states = pd.get_dummies(x['State'], drop_first = True)
"""One hot encoding is a technique that we use to represent 
categorical variables as numerical values in a machine learning model."""

# Drag the state column
x = x.drop('State',axis =1)

# concat the dummy data
x = pd.concat([x,states],axis =1)

#Splitting training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regg = LinearRegression()
regg.fit(x_train,y_train)

#predicting the test series result
y_pred = regg.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)