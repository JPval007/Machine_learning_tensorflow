import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense #probablemente va con mayuscula
import numpy as np
from matplotlib import pyplot
from scipy import io

def distribute_train_validation_split(validation_size=0.2):

    #all_images = os.listdir('./input_from_kaggle/train/')
    #random.shuffle(all_images)

    dog_Data = io.loadmat('/dogs/dog_data.mat') #list(filter(lambda image: 'dog' in image, all_images))
    cat_Data = io.loadmat('/cats/cat_data.mat') #list(filter(lambda image: 'cat' in image, all_images))

    split_num = int(len(dog_Data) - len(dog_Data) * validation_size)
    training_dogs = dog_Data[:split_num]
    validation_dogs = dog_Data[split_num:]
    training_cats = cat_Data[:split_num]
    validation_cats = cat_Data[split_num:]

    shutil.rmtree('./input_for_model')
    os.makedirs('./input_for_model/train/dogs/', exist_ok=True)
    os.makedirs('./input_for_model/train/cats/', exist_ok=True)
    os.makedirs('./input_for_model/validation/dogs/', exist_ok=True)
    os.makedirs('./input_for_model/validation/cats/', exist_ok=True)

    duplicate_in_dir(training_dogs, './input_for_model/train/dogs')
    duplicate_in_dir(validation_dogs, './input_for_model/validation/dogs')
    duplicate_in_dir(training_cats, './input_for_model/train/cats')
    duplicate_in_dir(validation_cats, './input_for_model/validation/cats')

def duplicate_in_dir(images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f'./input_from_kaggle/train/{image}', f'{destination}/{image}')

distribute_train_validation_split(0.25)


train_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)
validation_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)

train_iterator = train_imagedatagenerator.flow_from_directory(
    './input_for_model/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary')

validation_iterator = validation_imagedatagenerator.flow_from_directory(
    './input_for_model/validation',
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary')

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((4, 4)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((4, 4)),
    keras.layers.Conv2D(32, (5, 5), activation='relu'),
    keras.layers.MaxPool2D((8, 8)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])



#Before we start training we need to compile the model
model.compile(optimizer='sgd', loss='binary_cross_entropy', metrics=['accuracy'])
model.summary()



#Training the model
history = model.fit(train_iterator,
                    validation_data=validation_iterator,
                    steps_per_epoch=50,
                    epochs=100,
                    validation_steps=50)



#Visualizing the training result
def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



#Optimizing the model
    #Image augmentation
train_imagedatagenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')



#Creating a prediction
def load_and_predict():
    model = keras.models.load_model('dogs-vs-cats.h5')

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './input_test',
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    ids = []
    for filename in test_iterator.filenames:
        ids.append(int(filename.split('\\')[1].split('.')[0]))

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        predictions.append([ids[index], prediction[0]])
    predictions.sort()
    
    return predictions


predictions = load_and_predict()
df = pd.DataFrame(data=predictions, index=range(1, 12501), columns=['id', 'label'])
df = df.set_index(['id'])
df.to_csv('output_result.csv')
