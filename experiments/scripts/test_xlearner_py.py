#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
import numpy as np
import timeit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor

data = pd.read_csv("test_big_data.csv")
X = data.drop(columns=['treatment','outcome'])
Y = data['outcome']
T = data['treatment']


models = LinearRegression()
propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6)
X_learner = XLearner(models=models, propensity_model=propensity_model)


# Train S_learner
start = timeit.default_timer()
X_learner.fit(Y, T, X=X)
stop = timeit.default_timer()

print("S-learner time (Python):", stop - start)
# Estimate treatment effects on test data
X_te = X_learner.effect(X)
print("T-learner ATE:", np.mean(S_te))

