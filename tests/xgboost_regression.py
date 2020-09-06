#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:49:49 2020

@author: wmonteiro92
"""

from xmoai.setup.configure import generate_counterfactuals_regression
from sklearn_data.datasets import load_sample_from_dataset
from xgboost_data.models import train_ml_model

import numpy as np
from view_results import view_results

def train_regression(y_desired_index=1, verbose=False):
    """Train multiple datasets (regression problem) under a XGBoost implementation.

    :param y_desired_index: the index of the datasset to retrieve an instance
        to test. Default is 1.
    :type y_desired_index: Integer
    :param verbose: define the verbosity. Default is False.
    :type verbose: Boolean
    
    :return: an array containing the results found. Each row includes the 
        dataset used, the algorithm used, the Pareto front, the Pareto set
        (i.e. the counterfactual variables) and the multiobjective optimization
        algorithm responsible of each counterfactual, respectively.
    :rtype: np.array
    """
    
    datasets = ['boston', 'diabetes', 'california']
    algorithms = ['XGBRegressor']
    trained_models = []
    
    for dataset in datasets:
        if verbose:
            print(f'Retrieving the dataset {dataset}')
        
        # Setting the sample index to be returned from the database
        y_desired_index = 1
        
        # Loading the database (in order to train the model) and 
        # required additional metadata on the sample in order to generate
        # the contrafactuals
        X, y, X_current, y_desired, immutable_column_indexes, \
            upper_bounds, lower_bounds, y_acceptable_range, \
            categorical_columns, integer_columns = \
                load_sample_from_dataset(y_desired_index, dataset)
        
        # Training a ML model
        for algorithm in algorithms:
            if verbose:
                print(f'Training using {algorithm}')
            
            model = train_ml_model(X, y, algorithm)
            
            if verbose:
                print(f'Starting counterfactual generation.')
                
            pareto_front, pareto_set, pareto_algorithms = \
                generate_counterfactuals_regression(model, X_current, \
                    y_desired, immutable_column_indexes, y_acceptable_range, \
                    upper_bounds, lower_bounds, categorical_columns, \
                    integer_columns, n_gen=50, pop_size=300, \
                    max_changed_vars=10, verbose=verbose, seed=0)
                
            trained_models.append([dataset, algorithm, pareto_front,
                                   pareto_set, pareto_algorithms])
    
    return np.array(trained_models)

results = train_regression(verbose=True)
view_results(results)