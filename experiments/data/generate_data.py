#!/usr/bin/env python
# coding: utf-8

# REFERENCE: this code snippet is taken from one of Econml's example notebooks.
# TODO: change this to an easier script or a smaller example dataset (e.g., grad admission rate prediction on kaggle)

# Helper imports 
import numpy as np
from numpy.random import binomial, multivariate_normal, normal, uniform
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt


# In[27]:



# Define DGP
def generate_data(n, d, controls_outcome, treatment_effect, propensity):
    """Generates population data for given untreated_outcome, treatment_effect and propensity functions.
    
    Parameters
    ----------
        n (int): population size
        d (int): number of covariates
        controls_outcome (func): untreated outcome conditional on covariates
        treatment_effect (func): treatment effect conditional on covariates
        propensity (func): probability of treatment conditional on covariates
    """
    # Generate covariates
    X = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
    # Generate treatment
    T = np.apply_along_axis(lambda x: binomial(1, propensity(x), 1)[0], 1, X)
    # Calculate outcome
    Y0 = np.apply_along_axis(lambda x: controls_outcome(x), 1, X)
    treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
    Y = Y0 + treat_effect * T
    return (Y, T, X)


# In[28]:



# controls outcome, treatment effect, propensity definitions
def generate_controls_outcome(d):
    beta = uniform(-3, 3, d)
    return lambda x: np.dot(x, beta) + normal(0, 1)
treatment_effect = lambda x: (1 if x[1] > 0.1 else 0)*8
propensity = lambda x: (0.8 if (x[2]>-0.5 and x[2]<0.5) else 0.2)


# In[29]:


# DGP constants and test data
d = 5
n = 1000000
n_test = 250
controls_outcome = generate_controls_outcome(d)
X_test = multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n_test)
delta = 6/n_test
X_test[:, 1] = np.arange(-3, 3, delta)


# In[30]:


Y, T, X = generate_data(n, d, controls_outcome, treatment_effect, propensity)


# In[6]:


import pandas as pd
final_df = pd.concat([pd.DataFrame(X),pd.DataFrame(T),pd.DataFrame(Y)], axis=1)


# In[31]:


final_df.columns = ['var1', 'var2', 'var3', 'var4', 'var5', 'treatment', 'outcome']


# In[32]:


final_df.to_csv(r"D:\Downloads\test_big_data.csv")

