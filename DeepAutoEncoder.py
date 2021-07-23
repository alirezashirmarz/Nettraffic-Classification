# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 00:53:17 2021

@author: alireza
"""
""" Import Keras & Requirements """
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


inputs=keras.Input(shape=(23,))
dense=layers.Dense(10,activation='relu')
x=dense(inputs)
x=layers.Dense(5,activation='relu')(x)
outputs=layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

print(model.summary())




keras.utils.plot_model(model, "my_first_model.png")


print(inputs.shape)
