# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:31:24 2021

@author: sthan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('BMI.csv')

y = df.pop('Gender')

y = y.replace({'Male': 1, 'Female': 0})


reg = LogisticRegression()

reg.fit(df, y)

# Saving model to Disk
pickle.dump(reg, open('model.pkl', 'wb'))

#Loading model to Compare the results

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[70, 150]]))


