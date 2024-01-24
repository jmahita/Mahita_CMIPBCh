#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import catboost
from catboost import CatBoostRegressor as cbr
from sklearn.utils import resample
from sklearn.utils import shuffle
import xgboost


df1 = pd.read_csv(r'PT_IgG_abtiters_0_14_age.csv')

df1b = df1[["day_0_MFI_normalised","Fold_change"]]

lenth = len(df1b.columns)
print(lenth)

X_train = df1b.iloc[:, :lenth-1]
y_train = df1b.iloc[:, -1]

df2 = pd.read_csv(r'2022_abtiter_prediction_task_age.csv')
df2b = df2[["day_0_MFI_normalised"]]#"Fold_change"]]
subject = df2["subject_id"]

age = df2["Age"]
vacc = df2["infancy_vac"]
X_test = df2b.iloc[:,:]



cat_predictions = {}

cat_model = cbr(iterations=50, depth=3, learning_rate=0.05, loss_function='RMSE')

cat_feat_indices = np.where(X_train.dtypes != float)[0]

#Train model on training dataset
cat_model.fit(X_train, y_train,cat_features=cat_feat_indices)

y_pred = cat_model.predict(X_test)
#print(y_test)
print(subject)
#print(age)
print(X_test)
print(y_pred)

