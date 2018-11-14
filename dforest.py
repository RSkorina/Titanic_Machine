#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:19:15 2018

@author: clararichter
"""

# from estimator import Estimator
import scipy as sp
import numpy as np
from dtree import DecisionTree

class DecisionForest():
    
    def __init__(self, n_estimators, bootstrap=True, russells_method=False): #max_features
        self.bootstrap = bootstrap
        self.russells_method = russells_method
        self.n_estimators=n_estimators
        
    def fit(self, X, y):
        '''The common way is to, at each node, pick d features randomly and 
        and select that which maximizes the information gain for splitting.
        This is done for performance reasons, and it's particularly useful if the
        number of features in the dataset is very large. 
        [In the API of sklearns DecisionTreeClassifier class, this option is given 
        through the paramter 'splitter'. In the API of DecisionForest the attribute
        'max_features' specifies the number of features consider]
        However, Since the number of features of our dataset was limited to 14, we decided to
        do an exhaustive search of the features at each node'''
        self.estimators_ = []
        
        # Russell's method
        if self.russells_method:
            for i in range(self.n_estimators):
                size = int(X.shape[1]/1.5)
                idxs = np.random.choice(range(X.shape[1]), size, replace=False)
                samples = (X.T[idxs]).T
                tree = DecisionTree(max_depth=4)
                tree.fit(samples, y)
                self.estimators_.append(tree)
            return 
        
        # Standard method
        for i in range(self.n_estimators):
            # for some reason I don't know, we draw n samples with REPLACEMENT
            # this is what is called bootstrapping
            idxs = np.random.choice(range(X.shape[0]), X.shape[0], \
                                    replace=True if self.bootstrap else False)
            tree = DecisionTree(max_depth=4)
            tree.fit(X[idxs], y[idxs])
            self.estimators_.append(tree)
            
            
    def predict(self, X):
        predictions = np.empty((X.shape[0], self.n_estimators))
        for j in range(self.n_estimators):
            predictions[:,j] = self.estimators_[j].predict(X)
        
        return sp.stats.mode(predictions, axis=1)[0].flatten().astype(int)
    
    def score(self, X, y):
        return ( ( self.predict(X) == y ) * 1 ).sum()/y.shape[0]