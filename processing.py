import pandas as pd
import numpy as np

df = pd.read_csv(url,sep=';',encoding='latin1').drop(['Unnamed: 0'], axis=1)
df['Rooms'] = df['Rooms'].str.replace(',','.')
df['Rooms'] = pd.to_numeric(df['Rooms'], errors='coerce')
df_real = df.copy()

df['Rent'] = df['Rent'].apply(lambda x: np.log(x))
df['Total-Area'] = df['Total-Area'].apply(lambda x: np.log(x) if x > 1 else 1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

#Metrics
from sklearn.metrics import *

"""
First we'll try one hot encoding hood columns and then creating prediction models.
Also we'll work with raw numbers instead of grouping and later on do feature engineering.
"""
df_work = df.copy()
one_hot = pd.get_dummies(df_work['Hood'])
df_work = df_work.drop(['Hood', 'Street'], axis=1)
df_work = df_work.join(one_hot)
Y = df_work.Rent
X = df_work.drop(['Rent'], axis=1)

pickle = df_work.iloc[0:1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 577)

from xgboost import XGBRegressor

n = 1000 #max iter
nb = XGBRegressor(random_state=1362, objective ='reg:squarederror').fit(x_train, y_train)
nb.predict(x_test)
nb.score(x_test, y_test)

#Let's try changing the algorithm to ball_tree.
for i in range(2,10):
    scores = cross_val_score(nb, x_train, y_train, cv=i)
    print(scores.mean())

import pickle
model = nb

# save the model to disk
Pkl_Filename = "pickle_xgb_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(nb, file)