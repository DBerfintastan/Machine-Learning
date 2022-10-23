# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:58:05 2022

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

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

scaler.fit(x_test)
x_test=scaler.transform(x_test)

mlpc_model=MLPClassifier().fit(x_train,y_train)
mlpc_model.coefs_

#doğrusal için relu sınıflandırma için logistic
#solver adam çok büyük boyutlu verilerde kullanılır. lbfgs küçük verilerde kullanılır.

y_pred=mlpc_model.predict(x_test)
print("Accuracy:" ,accuracy_score(y_test, y_pred))

mlpc_params={"alpha":[0.1,0.001,0.03,0.0001,1,5],
             "hidden_layer_sizes":[(10,10),(100,100,100),(100,100),(3,5)]}

mlpc=MLPClassifier(solver="lbfgs", activation="logistic")
mlpc_cv_model=GridSearchCV(mlpc, mlpc_params, cv=10, n_jobs=-1,verbose=2).fit(x_train, y_train)
mlpc_cv_model.best_params_

#final model

mlpc_tuned_model=MLPClassifier(solver="lbfgs", alpha=5, hidden_layer_sizes= (100,100)).fit(x_train,y_train)
y_pred_tuned=mlpc_tuned_model.predict(x_test)
print("Tuned Accuracy:" ,accuracy_score(y_test, y_pred_tuned))