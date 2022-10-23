# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:31:54 2022

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
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=42)

cart_model=DecisionTreeClassifier().fit(x_train,y_train)
y_pred= cart_model.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

cart=DecisionTreeClassifier()
cart_params={"max_depth":[2,3,5,8],
             "min_samples_split":[2,3,5,10,20,50]}

cart_cv_model=GridSearchCV(cart, cart_params,cv=10,n_jobs=-1,verbose=2).fit(x_train, y_train)
cart_cv_model.best_params_

#final model
cart_tuned_model=DecisionTreeClassifier(max_depth=5,min_samples_split=20).fit(x_train,y_train)
y_pred_tuned=cart_tuned_model.predict(x_test)
print("Tuned Accuracy:" ,accuracy_score(y_test, y_pred_tuned))

