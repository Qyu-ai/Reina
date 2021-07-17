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

# Instantiate T learner
overall_model = LinearRegression()
S_learner = SLearner(overall_model=overall_model)
# Train S_learner

start = timeit.default_timer()
S_learner.fit(Y, T, X=X)
stop = timeit.default_timer()

print("S-learner time (Python):", stop - start)
S_te = S_learner.effect(X)
print("S-learner ATE:", np.mean(S_te))

