# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:30:31 2022

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

rf_model=RandomForestClassifier().fit(x_train,y_train)
y_pred=rf_model.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

rf=RandomForestClassifier()
rf_params={"n_estimators":[100,200,500,1000],
           "max_features":[3,5,7,8],
           "min_samples_split":[2,5,10,20]}

rf_cv_model=GridSearchCV(rf,rf_params,cv=10, n_jobs=-1,verbose=2).fit(x_train,y_train)
print("Best params:" ,rf_cv_model.best_params_)

#final model

rf_tuned_model=RandomForestClassifier(max_features= 8, min_samples_split= 2, n_estimators= 200).fit(x_train,y_train)
y_tuned_pred=rf_tuned_model.predict(x_test)
print("Tuned Accuracy:",accuracy_score(y_test, y_tuned_pred))

#degisken onem duzeyleri

feature_imp=pd.Series(rf_tuned_model.feature_importances_,index=x_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Degisken Onem Skorlari")
plt.ylabel("Degiskenler")
plt.title("Degisken Onem Duzeyleri")
plt.show()