import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from skrebate import SURF
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import time

# reading csv file and extracting class column to y.

data = pd.read_csv("Data/Train/fa_md_train_new.csv")
test_data = pd.read_csv("Data/Test/fa_md_test.csv")

"""
Features to be selected
"""

data['FA Corticospinal Tracts Both'] = (data['FA Corticospinal Tracts Left']+data['FA Corticospinal Tracts Right'])/2
data['MD Corticospinal Tracts Both'] = (data['MD Corticospinal Tracts Left']+data['MD Corticospinal Tracts Right'])/2
data = data.drop(['FA Corticospinal Tracts Left','FA Corticospinal Tracts Right','MD Corticospinal Tracts Left','MD Corticospinal Tracts Right'], axis=1)

test_data['FA Corticospinal Tracts Both'] = (test_data['FA Corticospinal Tracts Left']+test_data['FA Corticospinal Tracts Right'])/2
test_data['MD Corticospinal Tracts Both'] = (test_data['MD Corticospinal Tracts Left']+test_data['MD Corticospinal Tracts Right'])/2
test_data = test_data.drop(['FA Corticospinal Tracts Left','FA Corticospinal Tracts Right','MD Corticospinal Tracts Left','MD Corticospinal Tracts Right'], axis=1)

"""
Shuffling the training set
"""

data = data.sample(frac=1)
print(data)
"""
Creating Training and Testing Labels
"""
y = data.Research_Group
x = data.drop('Research_Group', axis=1)
y_TEST = test_data.Research_Group
x_TEST = test_data.drop(['Series_ID', 'Research_Group'], axis=1)


labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
y_TEST = labelencoder_Y.fit_transform(y_TEST)

sm = SMOTE(sampling_strategy='not majority', k_neighbors=40)
sm_TEST = SMOTE(sampling_strategy='not majority', k_neighbors=20)
X_smote, Y_smote = sm.fit_resample(x, y)
X_TEST_SMOTE, Y_TEST_SMOTE = sm_TEST.fit_resample(x_TEST, y_TEST)
print(X_smote.shape)
print(np.unique(Y_smote, return_counts=True))
print("Test data")
print(X_TEST_SMOTE.shape)
print(np.unique(Y_TEST_SMOTE, return_counts=True))

# sc = StandardScaler()
# X_smote = sc.fit_transform(X_smote)
# x_TEST = sc.fit_transform(x_TEST)
# X_TEST_SMOTE = sc.fit_transform(X_TEST_SMOTE)

rf_classifier_1000_en = make_pipeline(StandardScaler(),
                                      SURF(n_jobs=-1, n_features_to_select=50),
                                      RandomForestClassifier(n_jobs=-1, n_estimators=2000, max_features=None,
                                                             criterion='entropy', random_state=0))

# rf_classifier_1000_gini = make_pipeline(StandardScaler(),
#                                         SURF(n_jobs=-1, n_features_to_select=50),
#                                         RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_features=None,
#                                                                criterion='gini', random_state=0))
rf_classifier_1000_en.fit(X_smote, Y_smote)
preds_en = rf_classifier_1000_en.predict(X_TEST_SMOTE)
eval_score = accuracy_score(preds_en, Y_TEST_SMOTE)
print(preds_en)
print(Y_TEST_SMOTE)
print(eval_score)
