# ============================================================================
# MT3006 - LABORATORIO 4, PROBLEMA 2
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal de una capa para resolver un problema de regresión con el cual 
# pueda colocarse un precio adecuado para un carro usado en el Reino Unido.
# ============================================================================
import tensorflow as tf
import numpy as np
import pandas as pd # procesamiento de data, CSV I/O (pd.read_csv)
from scipy import io

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler 
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

#Distribución de los precios

sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))
sns.distplot(data.price)
plt.title("Distribución de los precios")
plt.show()

#Correlacion entre datos

plt.figure(figsize=(10, 5))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.title("Correlación entre datos")
plt.show()

#Entrenamiento de un modelo de predicción de precio
predict = "price"
data = data[["year","price","mileage","tax","mpg","engineSize"]]

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.4)


model = DecisionTreeRegressor()
hist = model.fit(xtrain,ytrain)
predictions = model.predict(xtest)

print("Exactitud del modelo \n")
print(model.score(xtest,predictions))

#Graficamos los datos
'''
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
'''

#Source material
#https://thecleverprogrammer.com/2021/08/04/car-price-prediction-with-machine-learning/


