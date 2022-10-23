# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:20:29 2022

@author: dberf
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score 
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,mean_squared_error,r2_score, roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore" , category=DeprecationWarning)
warnings.filterwarnings("ignore" , category=FutureWarning)


data=pd.read_csv("diabetes.csv")

y = data["Outcome"]
x = data.drop(["Outcome"], axis=1)


log_model=LogisticRegression(solver="liblinear").fit(x,y)
log_model.intercept_
log_model.coef_

"""
y_pred=log_model.predict(x)
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
"""

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=42)
y_pred=log_model.predict(x_test)
log_model=LogisticRegression(solver="liblinear").fit(x_train, y_train)

print(accuracy_score(y_test, y_pred))
print(cross_val_score(log_model,x_test,y_test,cv=10).mean())