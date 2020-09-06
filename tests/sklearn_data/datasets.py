# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:37:05 2020

@author: wmont
"""

from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes, \
    load_digits, load_iris, load_wine, fetch_california_housing
    
import numpy as np

def get_dataset(dataset_name='breast_cancer'):
    """Retrieve one of the standard datasets in sklearn.

    :param dataset_name: the dataset name to use from sklearn. Valid values are 
        `breast_cancer`, `digits`, `iris`, `wine` for classification and 
        `boston`, `diabetes` and `california` for regression. Default is `breast_cancer`.
    :type dataset_name: str
    
    :return: Five variables are returned. First is the dataset itself without
        the target values; second includes the target values; third has
        all the categorical columns; fourth has all the integer columns and
        the last informs if it is a classification problem (True) or a regression
        problem (False).
    :rtype: np.array, np.array, np.array, np.array, Boolean
    """
    
    if dataset_name == 'breast_cancer':
        # loading the dataset
        X, y = load_breast_cancer(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {}
        integer_columns = []
        is_classification = True
    elif dataset_name == 'digits':
        # loading the dataset
        X, y = load_digits(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {}
        integer_columns = list(range(64))
        is_classification = True
    elif dataset_name == 'iris':
        # loading the dataset
        X, y = load_iris(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {}
        integer_columns = []
        is_classification = True
    elif dataset_name == 'wine':
        # loading the dataset
        X, y = load_wine(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {}
        integer_columns = [4, 12]
        is_classification = True
    elif dataset_name == 'boston':
        X, y = load_boston(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {3: [0, 1]}
        integer_columns = [8, 9]
        is_classification = False
    elif dataset_name == 'diabetes':
        # loading the dataset
        X, y = load_diabetes(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {1: [ 0.05068012, -0.04464164]}
        integer_columns = []
        is_classification = False
    elif dataset_name == 'california':
        # loading the dataset
        X, y = fetch_california_housing(return_X_y=True)

        # informing categorical columns and their available values
        categorical_columns = {}
        integer_columns = [1, 4]
        is_classification = False
        
    return X, y, categorical_columns, integer_columns, is_classification

def get_instance_from_dataset(X, index, dataset_name='breast_cancer'):
    """Retrieve one of the instances from the dataset to generate the 
    counterfactuals.

    :param X: the input samples.
    :type X: np.array
    :param index: the index relative to the sample to be retrieved.
    :type index: Integer
    :param dataset_name: the dataset name to use from sklearn. Valid values are 
        `breast_cancer`, `digits`, `iris`, `wine` for classification and 
        `boston`, `diabetes` and `california` for regression. Default is `breast_cancer`.
    :type dataset_name: str
    
    :return: Six variables are returned. First is the instance from the dataset
        in reference to the index provided; second is the list of columns that
        cannot be modified; third and fourth are the upper and lower bounds for
        each variable, respectively; fifth includes the acceptable range to be
        considered with the desired target; sixth is the desired target (outcome).
    :rtype: np.array, np.array, np.array, np.array, np.array, Integer
    """
    
    # get a instance in the i-th row
    X_current = X[index, :].flatten()
    
    if dataset_name == 'breast_cancer':
        count_class = 2
        
        # define which columns must remain untouched
        immutable_column_indexes = []
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 1
        y_acceptable_range = np.array([1.0/count_class, 1.0])
    elif dataset_name == 'boston':
        # define which columns must remain untouched
        immutable_column_indexes = [1, 5]
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 30
        y_acceptable_range = np.array([y_desired * 0.98, y_desired * 1.02])
    elif dataset_name == 'diabetes':
        # define which columns must remain untouched
        immutable_column_indexes = [1, 4, 5, 6]
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 200
        y_acceptable_range = np.array([y_desired * 0.95, y_desired * 1.05])
    elif dataset_name == 'digits':
        count_class = 10
        
        # define which columns must remain untouched
        immutable_column_indexes = [2, 5, 10, 20, 30, 40, 50]
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 9
        y_acceptable_range = np.array([1.0/count_class, 1.0])
    elif dataset_name == 'iris':
        count_class = 3
        
        # define which columns must remain untouched
        immutable_column_indexes = []
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 2
        y_acceptable_range = np.array([1.0/count_class, 1.0])
    elif dataset_name == 'wine':
        count_class = 3
        
        # define which columns must remain untouched
        immutable_column_indexes = [7, 8, 9]
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 2
        y_acceptable_range = np.array([1.0/count_class, 1.0])
    elif dataset_name == 'california':
        # define which columns must remain untouched
        immutable_column_indexes = [0, 1, 2]
        
        # defining how much can we modify the input values
        upper_bounds = np.max(X, axis=0)
        lower_bounds = np.min(X, axis=0)
        
        # defining what are the tolerable output values
        y_desired = 1.5
        y_acceptable_range = np.array([y_desired * 0.95, y_desired * 1.05])
        
    return X_current, immutable_column_indexes, \
        upper_bounds, lower_bounds, y_acceptable_range, y_desired

def load_sample_from_dataset(index, dataset_name='breast_cancer'):
    """Retrieve one of the instances from the dataset to generate the 
    counterfactuals as well as other dataset metadata and multiobjective
    optimization (MOO) design space info relative to the sample.

    :param index: the index relative to the sample to be retrieved.
    :type index: Integer
    :param dataset_name: the dataset name to use from sklearn. Valid values are 
        `breast_cancer`, `digits`, `iris`, `wine` for classification and 
        `boston`, `diabetes` and `california` for regression. Default is `breast_cancer`.
    :type dataset_name: str
    
    :return: Ten variables are returned. First is the dataset itself without
        the target values; second includes the target values; third is the 
        instance from the dataset in reference to the index provided; fourth 
        is the desired target (outcome); fifth is the list of columns that
        cannot be modified; sixth and seventh are the upper and lower bounds for
        each variable, respectively; eigth includes the acceptable range to be
        considered with the desired target; ninth has all the categorical 
        columns and tenth has all the integer columns.
    :rtype: np.array, np.array, np.array, Integer, np.array,
        np.array, np.array, np.array, np.array, np.array
    """
    
    # get a dataset
    X, y, categorical_columns, integer_columns, \
        is_classification = get_dataset(dataset_name)
    
    # get a instance from the dataset in the i-th row (defined in index)
    # as well as its predicted output
    X_current, immutable_column_indexes, upper_bounds, lower_bounds, \
        y_acceptable_range, y_desired = get_instance_from_dataset(X, index, dataset_name)
        
    return X, y, X_current, y_desired, immutable_column_indexes, \
        upper_bounds, lower_bounds, y_acceptable_range, categorical_columns, \
        integer_columns