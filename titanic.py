#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:01:57 2018

@author: clararichter
"""

import numpy as np
import pandas as pd
import sklearn
from dtree import DecisionTree
from dforest import DecisionForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


class cleaner(object):
    def __init__(self):
        self.meanPrice = []
        
    def ageSet(self,row):
        if row.Age <= 12:
            age = 0
        elif row.Age <= 18:
            age = 1
        elif row.Age <= 50:
            age = 2
        elif row.Age > 50:
            age = 3
        else:
            age = 4
        return age

    def aloneSet(self,row):
        #TODO Create status for if have parent or child
        #If traveling alone then alone = 1

        alone = 0
        if int(row.SibSp) == 0 and int(row.Parch) == 0:
            alone = 1
        return alone

    def ticketPriceSet(self,row):
        index = row.Pclass
        if row.Fare > row.meanTicketPrice:
            ticketPrice = 1
        else:
            ticketPrice = 0
        return ticketPrice
        
    def titanicClean(self,df):
        #made id an index
        df = df.set_index('PassengerId')

        df = df.drop([])

        # replace age with child young adult elder
        # if age is unknown Elder
        df['AgeClass'] = df.apply(self.ageSet, axis = 1)

        #Create alone
        df['Alone'] = df.apply(self.aloneSet, axis = 1)
                

        #find mean of fare
        for i in np.unique(df['Pclass']):
            self.meanPrice.append(df.Fare[df['Pclass'] == int(i)].mean(axis = 0))
        self.meanPrice = [0] + self.meanPrice

        #create column of array out of mean array
        #need to do this b/c apply function is weird
        tempTicketMean = []
        for index, row in df.iterrows():
            tempTicketMean.append(self.meanPrice[row.Pclass])    
        
        #Create ticket Price Column
        df['meanTicketPrice'] = tempTicketMean
        df['ticketPrice'] = df.apply(self.ticketPriceSet, axis = 1)

        df.Embarked = df.Embarked.fillna(value = 'S')
        #Change female to int
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
        
        
        df['Embarked'] = df['Embarked'].map({'Q': 1, 'S': 0, 'C':2})
        #drop uneeded columns
        df = df.drop(['Name','Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'meanTicketPrice'],axis = 1)
        df = pd.get_dummies(data = df, columns=['Pclass', 'AgeClass', 'Embarked'])
        
        return df

    

def train_data():
    clean = cleaner()
    df = pd.read_csv('train.csv')
    df = clean.titanicClean(df)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y
    

def test_data():
    clean = cleaner()
    df = pd.read_csv('test.csv')
    df = clean.titanicClean(df)
    X = df.values
    y = pd.read_csv('gender_submission.csv')
    y = y.set_index('PassengerId')
    y = y.values
    y = y.flatten()
    return X, y
    

clfs = [SVC(C=1.0, kernel='rbf'), 
        LinearSVC(penalty='l2', C=1.0, max_iter=1000),
        LogisticRegression(),
        DecisionTreeClassifier(max_depth=4, splitter='best'),
        RandomForestClassifier(n_estimators = 3, max_depth=4),
        DecisionTree(max_depth=4),
        DecisionForest(n_estimators=5, bootstrap=True),
        DecisionForest(n_estimators=100, russells_method=True)]

X_train, y_train = train_data()
X_test, y_test = test_data()

for clf in clfs:
    clf.fit(X_train, y_train)
    print("%s: \t[%f/%f]" % (clf.__class__.__name__, clf.score(X_train, y_train), clf.score(X_test, y_test)) )
    if isinstance(clf, DecisionTreeClassifier):
        sklearn.tree.export_graphviz(clf, out_file='tree.dot')





