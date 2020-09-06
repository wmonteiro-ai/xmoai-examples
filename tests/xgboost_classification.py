#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:49:49 2020

@author: wmonteiro92
"""

from xmoai.setup.configure import generate_counterfactuals_classification_proba
from sklearn_data.datasets import load_sample_from_dataset
from xgboost_data.models import train_ml_model

import numpy as np
from view_results import view_results

def train_classification_proba(y_desired_index=1, verbose=False):
    """Train multiple datasets (classification problem) under a XGBoost implementation.

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
    datasets = ['breast_cancer', 'digits', 'iris', 'wine']
    algorithms = ['XGBClassifier']
    trained_models = []
    
    for dataset in datasets:
        if verbose:
            print(f'Retrieving the dataset {dataset}.')
        
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
                print(f'Training using {algorithm}.')
            
            model = train_ml_model(X, y, algorithm)
            
            if verbose:
                print(f'Starting counterfactual generation.')
                
            pareto_front, pareto_set, pareto_algorithms = \
                generate_counterfactuals_classification_proba(model, \
                    X_current, y_desired, immutable_column_indexes, \
                    y_acceptable_range, upper_bounds, lower_bounds, \
                    categorical_columns, integer_columns, n_gen=50, \
                    pop_size=100, max_changed_vars=5, verbose=verbose, seed=0)
                
            trained_models.append([dataset, algorithm, pareto_front,
                                   pareto_set, pareto_algorithms])
    
    return np.array(trained_models)

results_proba = train_classification_proba(verbose=True)
view_results(results_proba)