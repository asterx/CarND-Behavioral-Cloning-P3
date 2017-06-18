# coding=utf-8
import tensorflow as tf
from keras import models, optimizers
from keras.layers import core, convolutional, pooling
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn import model_selection

# Hyperparameters
LR = 1e-04

def net():
    model = models.Sequential()
    model.add(Convolution2D(16, 3, 3, subsample=(2,2), input_shape=(33, 100, 3), activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    #model.compile(optimizer=optimizers.Adam(lr=LR), loss='mean_squared_error')
    model.compile(optimizer=optimizers.Adam(lr=LR), loss='mse')
    return model
