#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:41:57 2021

@author: N
"""
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
# Load dataset
X_WA = read_csv('/Users/Documents/PostdocUW/proc_data/X_SCA.csv') 
Y_WA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_SCA.csv') 


X_WA = X_WA.drop(columns = ['Unnamed: 0'])
Y_WA = Y_WA.drop(columns = ['index','date'])

X_WA = X_WA[1:31]
Y_WA = Y_WA[1:31]

X_WA['SB_SCA'] = label_encoder.fit_transform(pd.cut(Y_WA.SB_SCA, 10, retbins=True)[0])

# Split-out validation dataset
array = X_WA.values
X = array[:,1:19]
y = array[:,20]

# Spot Check Algorithms
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=6, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
fig = pyplot.figure()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison for Central CA')
pyplot.show()

#predict
# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)


#fig.set_size_inches(10,10)
fig.savefig(r'/Users/Documents/PostdocUW/figure/ML_Central_CA_3.pdf',bbox_inches='tight', format='pdf', dpi=1000)
