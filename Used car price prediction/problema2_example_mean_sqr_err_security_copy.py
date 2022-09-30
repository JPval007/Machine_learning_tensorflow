# ============================================================================
# MT3006 - LABORATORIO 4, PROBLEMA 2
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal de una capa para resolver un problema de regresión con el cual 
# pueda colocarse un precio adecuado para un carro usado en el Reino Unido.
# ============================================================================
#https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd # procesamiento de data, CSV I/O (pd.read_csv)
from scipy import io

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
#from sklearn.utils import shuffle 



#Primero se importan los datos
data = pd.read_csv (r'toyota.csv')


#Los datos tienen 9 columnas y la columna 2 está lo que deseo predecir
print("\nDatos en bruto\n")
print(data)
data=pd.get_dummies(data,columns=['model'])
data=pd.get_dummies(data,columns=['transmission'])
data=pd.get_dummies(data,columns=['fuelType'])
print("\nDatos codificados\n")
print(data)
print("\nInformación de los datos\n")
print(data.info())

#Creamos "features"
dataset = data.values
#Lo que entra en el modelo (todo los datos)
X = dataset[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]
#Lo que quiero predecir (precios)
Y = dataset[:,1]


#Escalamos los datos para que estén entre 0 y 1
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#Split data into training set, validation set and test set
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.7)
#esto le dice a scikit-learn que el 30% del dataset será validación y pruebas

#Como solo se puede separar en dos lo repetimos para extraer el 50% para separar el de prueba del de validación
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#Para saber cómo se ven los arreglos se puede correr
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


#Queremos una capa oculta con función sigmoide y un nodo de salida con relu
#emplear pérdida de error cuadrático medio

#Construimos el modelo en keras
model = Sequential([
    Dense(32,activation='sigmoid',input_shape=(31,)),
    Dense(32,activation='sigmoid'),
    Dense(1,activation='relu')
    ])

#Compilamos el modelo y lo entrenamos
#valores ideales learning=0.8, momenum=0.0001, funciones de activación en relu todas
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0000971,momentum=0.0009),
              loss='mean_squared_error',
              metrics=['accuracy'])
#learning=0.001, momentum=0.001
hist = model.fit(X_train,Y_train,
                    batch_size=32,epochs=200,
                    validation_data=(X_val,Y_val))

#Evaluamos el modelo
print("\nPrueba del modelo\n")
model.evaluate(X_test, Y_test)[1]

#Graficamos los datos
plt.subplot(1,2,1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
