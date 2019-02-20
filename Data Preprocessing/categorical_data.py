# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# label encoding for X [Independent Variable]
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

# label encoding for y [Dependent Variable]
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)