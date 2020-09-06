# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:16:08 2020

@author: wmonteiro92
"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def build_regression(num_attributes):
    """Build a Tensorflow model for a regression problem.

    :param num_attributes: the number of attributes to be considered.
    :type num_attributes: Integer
    :param num_classes: the number of classes.
    :type num_classes: Integer
    
    :return: the trained Tensorflow model.
    :rtype: Object
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[num_attributes]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def build_classification_proba(num_attributes, num_classes):
    """Build a Tensorflow model for a classification problem.

    :param num_attributes: the number of attributes to be considered.
    :type num_attributes: Integer
    :param num_classes: the number of classes.
    :type num_classes: Integer
    
    :return: the trained Tensorflow model.
    :rtype: Object
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[num_attributes]),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
    return model

def train_ml_model(X, y, algorithm, random_state=0, verbose=True):
    """Train one dataset in Python.

    :param X: the input values.
    :type X: np.array
    :param y: the target values.
    :type y: np.array
    :param algorithm: the machine learning model to use. Allowed values are 
        `tf-regression`, `tf-classification-proba`.
    :type algorithm: str
    :param random_state: the seed. Default is 0.
    :type random_state: Integer
    :param verbose: sets the verbosity. Default is True.
    :type verbose: Boolean
    
    :return: the trained machine learning model.
    :rtype: Object
    """
    
    if algorithm == 'tf-regression':
        model = build_regression(X.shape[1])
    elif algorithm == 'tf-classification-proba':
        model = build_classification_proba(X.shape[1], y.shape[1])
                
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2,
              verbose=1 if verbose else 0)
    
    return model