# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:58:32 2022

@author: dberf
"""

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
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=42)

knn_model=KNeighborsClassifier().fit(x_train, y_train)
y_pred=knn_model.predict(x_test)

print("Accuracy:" , accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

knn=KNeighborsClassifier()
knn_params={"n_neighbors": np.arange(1,50)}
knn_cv_model=GridSearchCV(knn, knn_params,cv=10).fit(x_train, y_train)
knn_cv_model.best_score_
knn_cv_model.best_params_


#final model

knn_tuned=KNeighborsClassifier(n_neighbors=11).fit(x_train, y_train)
y_pred_tuned=knn_tuned.predict(x_test)
print("Tuned Accuracy: ",accuracy_score(y_test, y_pred_tuned))