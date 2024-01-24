#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import catboost
from catboost import CatBoostRegressor as cbr
from sklearn.utils import resample
from sklearn.utils import shuffle
import xgboost


df1 = pd.read_csv(r'2020_2021_monocytes_batchcorr.csv')

df1b = df1[["infancy_vac","biological_sex","race","FC_batchcorr"]]

lenth = len(df1b.columns)
print(lenth)

X_train = df1b.iloc[:, :lenth-1]
y_train = df1b.iloc[:, -1]

df2 = pd.read_csv(r'2022_cellfreq_predtask_age.csv')
#df2b = df2[["correct_age,"infancy_vac","biological_sex","race","percent_live_cell_day_0","fold_change"]]#"Fold_change"]]
subject_id = df2["subject_id"]
df2b = df2[["infancy_vac","biological_sex","race"]]
X_test = df2b.iloc[:,:].values



cat_predictions = {}

cat_model = cbr(iterations=50, depth=3, learning_rate=0.05, loss_function='RMSE')

cat_feat_indices = np.where(X_train.dtypes != float)[0]

#Train model on training dataset
cat_model.fit(X_train, y_train,cat_features=cat_feat_indices)

y_pred = cat_model.predict(X_test)
print(X_test)
print(y_pred)
