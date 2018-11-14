#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:00:35 2018

@author: clararichter
"""

import numpy as np

class Node():
    def __init__(self, X_y, max_depth, usedCol, criterion='gini'):
        self.children = {}
        self.X_y = X_y
        self.criterion = criterion
        self.max_depth = max_depth
        self.usedCol = usedCol
        self.feature = None
        self.majority = None
    
    def build_tree(self, depth):
        counts = np.bincount(list(self.X_y[:,-1]))
        self.majority = np.argmax(counts)
        I_p = self.impurity()
        
        if depth > self.max_depth:
            return
        if I_p == 0:
            return
        
        if self.X_y.shape[1] < 2:
            return
        
        # decide what feature to split at
        max_IG = 0
        split_feature = 0 
        # For columns not in the used columns
        for j in set(range(self.X_y.shape[1] - 1)) ^ set(self.usedCol):
            children = {}            
            IG = I_p
        
            for feature_val in np.unique(self.X_y[:,j]):
                
                child_samples = self.X_y[ self.X_y[:,j] == feature_val ]
                # child_samples = np.delete(child_samples, j, 1)
                child = Node(child_samples, self.max_depth, np.append(self.usedCol, j))
                IG -= (child.X_y.shape[0]/self.X_y.shape[0]) * child.impurity()
                children[feature_val] = child
            if IG > max_IG:
                split_feature = j
                self.children = children
                max_IG = IG   
        
        self.feature = split_feature

        for feature_val, child in self.children.items():
            child.build_tree( depth + 1 )
                
    def impurity(self):
        if self.criterion == 'gini':
            return self.gini()
        elif self.criterion == 'entropy':
            return self.entropy()
        else:
            raise KeyError
        
    def entropy(self):
        return 0
    
    def gini(self):
        impurity = 0
        for c in np.unique(self.X_y[:,-1]):
            impurity += ( (c == self.X_y[:,-1] * 1).sum()/self.X_y.shape[0] )**2
        return 1 - impurity
    
    
class DecisionTree():
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        
    def fit(self, X, y):        
        self.classes = np.unique(y)
        X_y = np.append(X, y.reshape((y.shape[0], 1)), 1)
        self.root = Node(X_y, self.max_depth, [])
        self.root.build_tree(0)
    
    
    
    def predict(self, X):
        predictions = []
        
        for sample in X:
            node = self.root
            while(len(node.children)) > 0:
                value = sample[node.feature]
                try:
                    node = node.children[value] 
                except:
                    break
            predictions.append(node.majority)
        return np.array(predictions)
    
    def score(self, X, y):
        return ((self.predict(X) == y)*1).sum()/y.shape[0]
    

