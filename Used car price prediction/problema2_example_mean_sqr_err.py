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
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
import missingno as msno
from sklearn.utils import shuffle 
from category_encoders import TargetEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
sns.set(rc = {'figure.figsize': (20, 20)})


#Primero se importan los datos
data = pd.read_csv (r'toyota.csv')

#Ordenamos los modelos en números
#crear el dataframe inicial
models = ('Auris','Avensis','Aygo','Camry','C-HR','Corolla',
          'GT86','Hilux','IQ','Land Cruiser','Prius','PROACE VERSO',
          'RAV4','Supra','Urban Cruiser','Verso','Verso-S','Yaris')
#Otra forma de extraer este array es usando:
#models = data.model.unique()
model_data = pd.DataFrame(models, columns=['Model'])

#Generar valores binarios usando get_dummies
dum_data = pd.get_dummies(model_data, columns=["Model"],prefix=["Model_is"])


#Combinar con el df original con la copia de valores
model_data = model_data.join(dum_data)
print("Codificado de datos")
print(model_data)
#Los datos tienen 9 columnas y la columna 2 está lo que deseo predecir
'''
#Creamos "features"
dataset = data.values
codification = model_data
#Lo que entra en el modelo (todo los datos)
X = dataset[:,[0,1,3,4,5,6,7,8]]
#Lo que quiero predecir (precios)
Y = dataset[:,2]

#Escalamos los datos para que estén entre 0 y 1
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#Split data into training set, validation set and test set
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
#esto le dice a scikit-learn que el 30% del dataset será validación y pruebas

#Como solo se puede separar en dos lo repetimos para extraer el 50% para separar el de prueba del de validación
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#Para saber cómo se ven los arreglos se puede correr
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


#Queremos una capa oculta con función sigmoide y un nodo de salida con relu
#emplear pérdida de error cuadrático medio

#Construimos el modelo en keras
model = Sequential([
    Dense(1,activation='sigmoid', input_shape=(9)),
    Dense(1,activation='relu')
    ])
'''

