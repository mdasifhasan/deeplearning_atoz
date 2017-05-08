import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# SOM
from minisom import MiniSom
som = MiniSom(X, y, 15)