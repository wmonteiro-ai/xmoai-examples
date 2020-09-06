# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:16:08 2020

@author: wmonteiro92
"""

from lightgbm import LGBMClassifier, LGBMRegressor

def train_ml_model(X, y, algorithm, random_state=0):
    """Train one dataset in Python.

    :param X: the input values.
    :type X: np.array
    :param y: the target values.
    :type y: np.array
    :param algorithm: the machine learning model to use. Allowed values are 
        `LGBMClassifier` and `LGBMRegressor`.
    :type algorithm: str
    :param random_state: the seed. Default is 0.
    :type random_state: Integer
    
    :return: the trained machine learning model.
    :rtype: Object
    """
    
    if algorithm == 'LGBMClassifier':
        model = LGBMClassifier(random_state=random_state)
    elif algorithm == 'LGBMRegressor':
        model = LGBMRegressor(random_state=random_state)
        
    model.fit(X, y)
    return model