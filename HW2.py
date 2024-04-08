#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:42:20 2024

@author: solauphoenix(Salih KIZILIŞIK)
"""
#%%Libraries Reference:https://www.datacamp.com/tutorial/random-forests-classifier-python
import time  # For timing model training
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
#For String encoding
from sklearn.preprocessing import LabelEncoder

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
#import graphviz
#%%Data Preprocessing
df = pd.read_csv('HeartDiseaseDataset.csv')
print('Data frame shape:', df.shape)

#Tune the pandas for showing all columns
pd.options.display.max_columns = None

#Mapping
df['ExerciseAngina']=df['ExerciseAngina'].map({'N':0,'Y':1})

#Encoding the categorical string values
categ=['Sex','ChestPainType','RestingECG','ST_Slope']
le = LabelEncoder()
df[categ] = df[categ].apply(le.fit_transform)

#Showing all Columns
print(df.head())
#%%Data Splitting
X=df[['Age','Sex','ChestPainType','RestingBP','Cholesterol' ,'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',  'Oldpeak', 'ST_Slope']]
y=df[['HeartDisease']]

#Alternative Solution

## Split the data into features (X) and target (y)
#X = bank_data.drop('y', axis=1)
#y = bank_data['y']


#Transform y to numpy array
y= y.to_numpy()

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%Model Definition
rf = RandomForestClassifier()
start_time = time.time()
rf.fit(X_train, y_train.ravel())#Transforming y to 1D and run the model
end_time = time.time()

#Testing
y_pred = rf.predict(X_test)

# Reporting how long it takes to fit the model
print('Total time to fit the model:', end_time - start_time)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%% Visualization it uses python-grapviz if you dont have comment this section.
# Export the first three decision trees from the forest

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=10, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     display(graph)
#%%