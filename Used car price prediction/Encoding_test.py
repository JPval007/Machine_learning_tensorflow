# ============================================================================
# MT3006 - LABORATORIO 4, PROBLEMA 2
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal de una capa para resolver un problema de regresión con el cual 
# pueda colocarse un precio adecuado para un carro usado en el Reino Unido.
# ============================================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd # procesamiento de data, CSV I/O (pd.read_csv)
from scipy import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

#Primero se importan los datos
data = pd.read_csv (r'toyota.csv')
print("\nDatos en bruto\n")
print(data)
data=pd.get_dummies(data,columns=['model'])
print("\nDatos codificados\n")
print(data)
print("\nInformación de los datos\n")
print(data.info())

data = data.values
print(data[1,:])
