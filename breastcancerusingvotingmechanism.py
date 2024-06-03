# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

#"""LOADING DATA FROM CSV FILE"""

data=pd.read_csv("dataR2.csv")

#"""TO GET INFORMATION ABOUT VARIOUS PARAMETERS"""

data.describe()

data.head()

data['Classification'].value_counts()

X=data.drop(columns='Classification',axis=1)
Y=data['Classification']

print(X)
print(Y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
standard_data=scaler.transform(X)

X=standard_data
Y=data['Classification']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=0.2)

# Initialize individual models
from sklearn.ensemble import RandomForestClassifier
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)
model4 = RandomForestClassifier()

# Initialize voting classifier with soft voting
voting_clf = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3),('rf',model4)], voting='soft')

# Train the voting classifier
voting_clf.fit(X_train, Y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

input_data = (48,23.15,70,2.07,0.46,8.8,9.7,7.99,417)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = voting_clf.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have Breast Cancer')
if (prediction[0]==1):
  print('The person has Breast Cancer')

#name of model is voting_clf
pickle.dump(voting_clf,open("breastcancerusingvotingmechanism.pkl","wb"))


