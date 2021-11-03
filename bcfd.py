import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('project.csv')
print(data)
data=data.drop('Id',axis=1)
data=data.drop("CITY",axis=1)
en = LabelEncoder()
catCols = ['Married/Single','House_Ownership','Car_Ownership','Profession','STATE']
for cols in catCols:
    data[cols] = en.fit_transform(data[cols])
Y = data["Risk_Flag"]
X = data.drop("Risk_Flag",axis = 1)
X_train_full, X_valid_full, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train_full, Y_train)
print(lr.predict([[1303834,23,3,1,2,0,33,13,3,13]]))