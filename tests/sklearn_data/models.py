# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:16:08 2020

@author: wmonteiro92
"""

from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_ml_model(X, y, algorithm, random_state=0):
    """Train one dataset in Python.

    :param X: the input values.
    :type X: np.array
    :param y: the target values.
    :type y: np.array
    :param algorithm: the machine learning model to use. Allowed values are 
        `AdaBoostClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`,
        `RandomForestClassifier`, `DecisionTreeClassifier`, `NuSVC`,
        `LinearSVC`, `RidgeClassifier`, `SGDClassifier`,
        `LogisticRegression`, `KNeighborsClassifier`, `AdaBoostRegressor`,
        `ExtraTreesRegressor`, `GradientBoostingRegressor`, `RandomForestRegressor`,
        `DecisionTreeRegressor`, `NuSVR`, `LinearSVR`,
        `Ridge`, `SGDRegressor`, `Lasso`,
        `KNeighborsRegressor`, `RadiusNeighborsRegressor`.
    :type algorithm: str
    :param random_state: the seed. Default is 0.
    :type random_state: Integer
    
    :return: the trained machine learning model.
    :rtype: Object
    """
    
    if algorithm == 'AdaBoostClassifier':
        model = AdaBoostClassifier(random_state=random_state)
    elif algorithm == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(random_state=random_state)
    elif algorithm == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(max_depth=3, random_state=random_state)
    elif algorithm == 'RandomForestClassifier':
        model = RandomForestClassifier(max_depth=2, random_state=random_state)
    elif algorithm == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(random_state=random_state)
    elif algorithm == 'NuSVC':
        model = NuSVC(random_state=random_state)
    elif algorithm == 'LinearSVC':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', LinearSVC(random_state=random_state, max_iter=10000))])
    elif algorithm == 'RidgeClassifier':
        model = RidgeClassifier(random_state=random_state)
    elif algorithm == 'SGDClassifier':
        model = SGDClassifier(loss='modified_huber', random_state=random_state)
    elif algorithm == 'LogisticRegression':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', LogisticRegression(random_state=random_state))])
    elif algorithm == 'KNeighborsClassifier':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', KNeighborsClassifier(radius=2.0))])
    elif algorithm == 'AdaBoostRegressor':
        model = AdaBoostRegressor(random_state=random_state)
    elif algorithm == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor(random_state=random_state)
    elif algorithm == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(max_depth=3, random_state=random_state)
    elif algorithm == 'RandomForestRegressor':
        model = RandomForestRegressor(max_depth=2, random_state=random_state)
    elif algorithm == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(random_state=random_state)
    elif algorithm == 'NuSVR':
        model = NuSVR()
    elif algorithm == 'LinearSVR':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', LinearSVR(random_state=random_state, max_iter=10000))])
    elif algorithm == 'Ridge':
        model = Ridge(random_state=random_state)
    elif algorithm == 'SGDRegressor':
        model = SGDRegressor(random_state=random_state)
    elif algorithm == 'Lasso':
        model = Lasso(random_state=random_state)
    elif algorithm == 'KNeighborsRegressor':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', KNeighborsRegressor())])
    elif algorithm == 'RadiusNeighborsRegressor':
        model = Pipeline([('scaler', StandardScaler()), \
                          ('model', RadiusNeighborsRegressor())])
        
    model.fit(X, y)
    return model