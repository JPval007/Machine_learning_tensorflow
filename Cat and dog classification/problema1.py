# ============================================================================
# MT3006 - LABORATORIO 4, PROBLEMA 1
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar un
# perceptrón simple para encontrar un modelo que permita clasificar imágenes 
# de perros (0) y gatos (1). 
# ============================================================================
import tensorflow as tf
from tensorflow.keras.layers import Dense #probablemente va con mayuscula
import numpy as np
from matplotlib import pyplot
from scipy import io

#Parametros
learning_rate = 0.01
momentum = 0.9
training_epochs = 200 
batch_size = 100 
display_step = 1 

# Se cargan los archivos mat con la data
dogData = io.loadmat('dog_data.mat')
catData = io.loadmat('cat_data.mat')

# Se extrae la data y se re-dimensionan las imágenes como vectores columna
dogWave = dogData['dog_wave']
catWave = catData['cat_wave']

# Rearreglo aleatorio de las columnas
np.random.shuffle(np.transpose(dogWave))
np.random.shuffle(np.transpose(catWave))

# Se visualiza un ejemplo de cada una de las dos categorías
dogExample = dogWave[:, 9];
catExample = catWave[:, 9];
dogExample = np.reshape(dogExample, (32,32))
catExample = np.reshape(catExample, (32,32))
pyplot.subplot(121)
pyplot.imshow(dogExample, cmap = 'gray', vmin = 0, vmax = np.amax(dogExample))
pyplot.subplot(122)
pyplot.imshow(catExample, cmap = 'gray', vmin = 0, vmax = np.amax(catExample))
pyplot.show()

# Se crean los sets de entrenamiento y de prueba, junto con las etiquetas
trainSet = np.transpose(np.concatenate((dogWave[:,:60], catWave[:,:60]), axis = 1))
valSet = np.transpose(np.concatenate((dogWave[:,60:80], catWave[:,60:80]), axis = 1))
trainLabels = np.repeat(np.array([0, 1]), 60)
valLabels = np.repeat(np.array([0, 1]), 20)

# *****************************************************************************
# DEFINA EL MODELO AQUÍ
# *****************************************************************************
#layer2 = Dense(1,activation='relu')
layer1 = Dense(1,activation='sigmoid')
model = tf.keras.Sequential([layer1])
optimizer = tf.keras.optimizers.SGD(learning_rate,momentum)
model.compile(loss='binary_crossentropy',optimizer = tf.keras.optimizers.SGD(learning_rate=0.005,momentum=0.68),metrics=['accuracy'])
#valor ideal momentum=0.64

# *****************************************************************************
# ENTRENE EL MODELO AQUÍ
# *****************************************************************************
#           x           y           validation data             epochs
history = model.fit(trainSet,trainLabels,validation_data = (valSet,valLabels),epochs=200)
#model.predict(va)
#en model.predict hay que poner los datos de prueba para probar el algoritmo


# *****************************************************************************
# Obtenemos los pesos del modelo
# *****************************************************************************
weights = model.get_weights() #devuelve una lista de arrays de numpy
weights = weights[0] #el otro termino de la lista es el bias
#Solo guardo el array de 1024 elementos
#Reordenamos en un array de 32x32
weights_shrink = np.reshape(weights, (32,32))
#Graficamos como imagen
pyplot.imshow(weights_shrink, cmap = 'gray', vmin = 0, vmax = np.amax(weights_shrink))
pyplot.title('Pesos del modelo')
pyplot.show()

# Se evalúa el modelo para encontrar la exactitud de la clasificación (tanto 
# durante el entrenamiento y la validación)
_, train_acc = model.evaluate(trainSet, trainLabels, verbose = 0)
_, val_acc = model.evaluate(valSet, valLabels, verbose = 0)
print('Train: %.3f, Test: %.3f' % (train_acc, val_acc))

# Se grafica la evolución de la pérdida durante el entrenamiento y la 
# validación
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# Se grafica la evolución de la exactitud durante el entrenamiento y la 
# validación
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# Se obtienen las predicciones del modelo para el set de validación
yhat = model.predict(valSet)
# Se obtiene la matriz de confusión para el set de validación
confusion = tf.math.confusion_matrix(labels = valLabels, predictions = yhat)
print(confusion)
#
##
#33
##
#
