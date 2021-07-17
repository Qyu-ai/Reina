#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
import numpy as np
import timeit

data = pd.read_csv("test_big_data.csv")
X = data.drop(columns=['treatment','outcome'])
Y = data['outcome']
T = data['treatment']

models = LinearRegression()
T_learner = TLearner(models=models)
# Train T_learner

start = timeit.default_timer()
T_learner.fit(Y, T, X=X)
stop = timeit.default_timer()

print("T-learner time (Python):", stop - start)

T_te = T_learner.effect(X)
print("T-learner ATE:", np.mean(S_te))

