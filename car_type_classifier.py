"""
Case Studies in Data Analytics-Assignment 1

Author: Saikrishna Javvadi

Script to train the car type classifier using mobilenet.

References : https://www.tensorflow.org/tutorials/images/transfer_learning
             https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
             https://analyticsindiamag.com/a-practical-guide-to-implement-transfer-learning-in-tensorflow/
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def car_type_classifier():
    img_size = 160
    img_shape = (img_size, img_size, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Freezing the base model
    model = tf.keras.Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['acc'])

    model.summary()
    training_data = ImageDataGenerator(rescale=1 / 255,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.4,
                                       fill_mode='nearest')

    training_data_generator = training_data.flow_from_directory(
        './training_data',
        target_size=(img_size, img_size),  # Shaping the train image to requisite size
        batch_size=64,
        class_mode='binary')

    validation_data = ImageDataGenerator(rescale=1 / 255)
    validation_data_generator = validation_data.flow_from_directory(
        './validation_data',
        target_size=(img_size, img_size),
        batch_size=64,
        class_mode='binary')

    history = model.fit_generator(
        training_data_generator,
        epochs=25,
        validation_data=validation_data_generator,
        verbose=1)

    return model, history


if __name__ == '__main__':
    model, history = car_type_classifier()
    model.save('mobilenet_cars.h5')

    # Plot Learning Curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1.1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 8.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
