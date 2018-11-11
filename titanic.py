#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:01:57 2018

@author: clararichter
"""

import numpy as np
import pandas as pd
from dtree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import tree

#first GO Titanic Clean
#Could make Class and easier
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
        return df.drop(['Name','Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'meanTicketPrice'],axis = 1)

def one_hot_transform(X, y):
    enc = preprocessing.OneHotEncoder(categorical_features='auto')
    print(enc)
    enc.fit(X) 
    enc.transform(X)
    enc.tranform(y)
    

def train(classifier):
    clean = cleaner()
    df = pd.read_csv('train.csv')
    df = clean.titanicClean(df)
    
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    classifier.fit(X, y)
    
    print(((classifier.predict(X) == y)*1).sum()/y.shape[0])


def test_acc(classifier):
    clean = cleaner()
    df = pd.read_csv('test.csv')
    
    df = clean.titanicClean(df)

    X = df.values
    
    y = pd.read_csv('gender_submission.csv')
    y = y.set_index('PassengerId')
    y = y.values
    y = y.flatten()
    
    print(((classifier.predict(X) == y)*1).sum()/y.shape[0])

# clf = DecisionTreeClassifier(max_depth=4, splitter='random')
clf = RandomForestClassifier(n_estimators = 3, max_depth=4)
train(clf)
# tree.export_graphviz(clf, out_file='tree.dot')

test_acc(clf)


