# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:14:02 2022

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
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore" , category=DeprecationWarning)
warnings.filterwarnings("ignore" , category=FutureWarning)


data=pd.read_csv("diabetes.csv")

y = data["Outcome"]
x = data.drop(["Outcome"], axis=1)
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=42)

lgbm_model=LGBMClassifier().fit(x_train,y_train)
y_pred= lgbm_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

lgbm=LGBMClassifier()
lgbm_params={"learning_rate":[0.1,0.01,0.001],
             "n_estimators":[200,500,100],
             "max_depth":[1,2,35,8]}
lgbm_cv_model=GridSearchCV(lgbm, lgbm_params, n_jobs=-1, verbose=2).fit(x_train,y_train)
print("Best params:" , lgbm_cv_model.best_params_)

#final model
lgbm_tuned_model=LGBMClassifier(learning_rate= 0.01, max_depth= 8, n_estimators= 100).fit(x_train,y_train)
y_pred_tuned=lgbm_tuned_model.predict(x_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))



#degisken onem duzeyleri
feature_imp=pd.Series(lgbm_tuned_model.feature_importances_,index=x_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Degisken Onem Skorlari")
plt.ylabel("Degiskenler")
plt.title("Degisken Onem Duzeyleri")
plt.show()
