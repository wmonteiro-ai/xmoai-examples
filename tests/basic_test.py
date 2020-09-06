#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from xmoai.setup.configure import generate_counterfactuals_classification_proba

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# seed
random_state = 0

# getting a dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=random_state)

# training a machine learning model
clf = RandomForestClassifier(max_depth=2, random_state=random_state)
clf.fit(X_train, y_train)

# getting an individual (X_original), its original prediction (y_original) and
# the desired output (y_desired)
index = 0
X_original = X_test[0,:].reshape(1, -1)
y_original = clf.predict(X_original)
y_original_proba = clf.predict_proba(X_original)
y_desired = 1

print(f'The original prediction was {y_original} with probabilities {y_original_proba}')
print(f'We will attempt to generate counterfactuals where the outcome is {y_desired}.')

# generating counterfactuals
immutable_column_indexes = [2] # let's say we can't change the last column
categorical_columns = {} # there are no categorical columns
integer_columns = [] # there are no columns that only accept integer values
y_acceptable_range = [0.5, 1.0] # we will only accept counterfactuals with the predicted probability in this range

upper_bounds = np.array(X_train.max(axis=0)*0.8) # this is the maximum allowed number per column
lower_bounds = np.array(X_train.min(axis=0)*0.8) # this is the minimum allowed number per column.
# you may change the bounds depending on the needs specific to the individual being trained.

# running the counterfactual generation algorithm
front, X_generated, algorithms = generate_counterfactuals_classification_proba(clf,
                          X_original, y_desired, immutable_column_indexes,
                          y_acceptable_range, upper_bounds, lower_bounds,
                          categorical_columns, integer_columns, n_gen=20,
                          pop_size=30, max_changed_vars=3, verbose=False, 
                          seed=random_state)

"""
Outputs:
    X_generated: the counterfactuals generated. Each row is a counterfactual.
        Each column has the same layout of X_train and X_test.
    front: the Pareto front approximation for each row in X_generated.
        The first column represents the difference between 100% and the 
        probability of the desired class according to the trained machine
        learning model. For regression problems it is the difference between
        y_desired and the y found by each counterfactual. The closer to zero,
        the better. The second column represents the distance between the
        attribute values of X_original and the counterfactual. The closer
        to zero, the better. The third column represents the number of changed
        variables. The closer to zero, the better.
    algorithms: represents the multiobjective optimization algorithm in charge
        of creating each counterfactual.
"""