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
from matplotlib import pyplot
from scipy import io

#Primero se importan los datos
df = df = pd.read_csv (r'toyota.csv')

#Ordenamos los modelos en números
#crear el dataframe inicial
models = ('Auris','Avensis','Aygo','Camry','C-HR','Corolla',
          'GT86','Hilux','IQ','Land Cruiser','Prius','PROACE VERSO',
          'RAV4','Supra','Urban Cruiser','Verso','Verso-S','Yaris')
model_df = pd.DataFrame(models, columns=['Model'])

#Generar valores binarios usando get_dummies
dum_df = pd.get_dummies(model_df, columns=["Model"],prefix=["Model_is"])


#Combinar con el df original con la copia de valores
model_df = model_df.join(dum_df)
print(model_df)




