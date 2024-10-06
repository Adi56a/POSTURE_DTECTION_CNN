import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

STANDING_DIR = 'archive\standing'  
SITTING_DIR = 'archive\sitting'    


IMG_HEIGHT = 224                      
IMG_WIDTH = 224                                      
BATCH_SIZE = 32
EPOCHS = 1


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # i have only provided 2 neurons at output layer
                                                    # because of two parameter considered

    return model

# Load and preprocess data
def load_data():
    standing_images = [os.path.join(STANDING_DIR, img) for img in os.listdir(STANDING_DIR)]
    sitting_images = [os.path.join(SITTING_DIR, img) for img in os.listdir(SITTING_DIR)]
    
    images = []
    labels = []

    for img_path in standing_images:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(0) # here i used 0 for the standing class 

    for img_path in sitting_images:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(1)  # Class 1 for sitting

    return np.array(images), np.array(labels)


def train_model():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3)) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=BATCH_SIZE)

    # in my next file you will see the no direct import from this file but still model will work 
    model.save('movement_classifier.h5')   

    
    val_accuracy = history.history['val_accuracy'][-1]
    print(f'Validation Accuracy: {val_accuracy:.2f}')

if __name__ == "__main__":
    train_model()
