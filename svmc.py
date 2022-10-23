# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:42:35 2022

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

svm_model=SVC().fit(x_train, y_train)
y_pred=svm_model.predict(x_test)

print("Accuracy:" , accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

svm=SVC()

svm_params={"C": np.arange(1,10), "kernel":["linear","rbf"]}
svm_cv_model=GridSearchCV(svm, svm_params,cv=5, n_jobs=-1, verbose=2).fit(x_train, y_train)
svm_cv_model.best_score_
svm_cv_model.best_params_


#final model

svm_tuned=SVC(C=2,kernel="linear").fit(x_train, y_train)
y_pred_tuned=svm_tuned.predict(x_test)
print("Tuned Accuracy: ",accuracy_score(y_test, y_pred_tuned))