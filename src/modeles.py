# -*- coding: utf-8 -*-
"""
modeles used for training
"""


import tensorflow as tf
from losses import correlate, mae_cor


def create_model(
        name:str = "CNN_simple",
        filters= 16,
        activation = "sigmoid",
        input_shape = (2001,4),
        loss = mae_cor,
        metrics = ["mae", correlate],
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        ):
    """
    return compiled model

    :param str name: name of the architecture
    :param input_shape: input shape as iterable
    :param loss: loss as string (keras function) or object (custom function)
    :param metrics: iterable of string (keras function) and/or objects (custom function)
    :param optimizer: keras optimizer
    """


    if name == "CNN_simple5H":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=10,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
    
    
    elif name == "Chemical_5H":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(256, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
    
    else:
        raise ValueError(f"unknown model name {name}")

