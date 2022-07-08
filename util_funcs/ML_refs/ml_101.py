## NB: this file isn't really for running
#       more for reference, pulling obvious bits together in one place

## ALL THE IMPORTS IN LWB ML AND DL

# FOR JUPYTER NOTEBOOKS ONLY:
# %load_ext autoreload
# %autoreload 2

# import fundamentals
# import visualisation tools
# import preprocessing
# import scikit-learning modelling
# import model assessment
# import sklearn.metrics
# import pipelines

import math
import random
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import plotly.graph_objects as go

# preprocessing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# scikit modelling
from sklearn.linear_model import LinearRegression # explicit class import from module
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC

# model assessment
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

# sklearn metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# pipelines etc
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.metrics import make_scorer, mean_squared_log_error
# import graphviz
from sklearn.tree import export_graphviz
from xgboost import XGBRegressor

# get diagram with pipeline name
from sklearn import set_config; set_config(display='diagram')

# import selected parts of tensorflow tf
import tensorflow as tf
# tensorflow yellow underlines don't stop it working...
from tensorflow.keras import datasets
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, SimpleRNN, LSTM, GRU
from tensorflow.keras.metrics import Accuracy, Precision

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from tensorflow.keras.preprocessing import image_dataset_from_directory

# padding with tf / keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking

# ############################## ###############################
#       Loading data
# ############################## ###############################
#
#
#
#
#
#
#

data = pd.read_csv('path_to_csv_file.csv')


# ############################## ###############################
#       Exploratory Data Analysis and Data Cleaning
# ############################## ###############################
#
#
#
#
#
#
#

data.shape
data.head(15)
data.tail()
data.describe() # min, max, mean, count etc
data.info()

data.columns # see column names

data[['feature']].value_counts()
data['feature'].value_counts().sort_values()

data.Feature.unique() # to show uniques

# ############################## ###############################
#       Data Cleaning
# ############################## ###############################
#
#
#
#
#
#
#

#       Duplicates
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# count duplicates
duplicate_count = data.duplicated().sum()
duplicate_count

# drop all duplicates
data = data.drop_duplicates()


#       Missing data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# list empty values as a percentage by feature
empty = data.isnull().sum().sort_values(ascending=False);
empty / len(data)

#       Replacing values
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data.FeatureName.replace(np.nan, "NoG", inplace=True) #Replace NaN with...
data[['feature']] = data[['feature']].replace(
    np.nan, data['feature'].median()) # replace with median


# ############################## ###############################
#       Data Visualisation
# ############################## ###############################
#
#
#
#
#
#
#

# simple plots
plt.plot(some_data_or_df = data);

# chart features
plt.xlabel('a')
plt.ylabel('b')


# boxplot
data.boxplot(); # good for seeing outliers

# histograms
data.hist('feature_a'); # pd histogram
sns.histplot(data=data,x="feature",bins=30);
sns.distplot(x=data['feature'], kde = False);

# scatter plots
sns.scatterplot(data=data, x='column_name_x', y='column_name_y');
plt.scatter(data['column_a'], data['column_b'], color = 'red');
# for showing cyclical values
data.plot.scatter('sin_MoSold','cos_MoSold').set_aspect('equal');

# very cool 3d, interactive plot
import plotly.graph_objects as go
# for use in a Jupyter notebook...?
Z, range_a, range_b = 50,50,50
surface = go.Surface(x=range_a, y=range_b, z=Z)
scatter = go.Scatter3d(x=data['column_a'], y=data['column_b'],
                       z=data['loss_history'], mode='markers')
fig = go.Figure(data=[surface, scatter])

fig.update_layout(title='Loss Function', autosize=False,
                  width=1000, height=800)
fig.show()

# correlation heatmap
corr = data.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= "YlGnBu");
# show the correlation between column pairs in a dataframe.
corr_df = corr.unstack().reset_index()
corr_df.columns = ['feature_1','feature_2', 'correlation'] # rename columns
corr_df.sort_values(by="correlation",ascending=False, inplace=True)
corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']]
corr_df

# ############################### ###############################
#       Feature selection
# ############################### ###############################
#
#
#
#
#
#
#

# drop columns
data = data.drop(columns = 'feature')

# Only keep numerical columns and raws without NaN
data = data.select_dtypes(include=np.number).dropna()

# select by column name
data = data[['column_one','column_two']]

y = data['column_name']
X = data.drop(columns = 'column_name')

# correlation heatmap
corr = data.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= "YlGnBu");
# show the correlation between column pairs in a dataframe.
corr_df = corr.unstack().reset_index()
corr_df.columns = ['feature_1','feature_2', 'correlation'] # rename columns
corr_df.sort_values(by="correlation",ascending=False, inplace=True)
corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']]
corr_df

# Feature Permutation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# To rank features by order of importance.

X_ = X[:500]
y_ = y[:500]
lin_model = LinearRegression().fit(X_, y_) # Fit model on sub_sample

permutation_score = permutation_importance(lin_model, X_, y_, n_repeats=10)
importance_df = pd.DataFrame(np.vstack((X.columns,permutation_score.importances_mean)).T)
importance_df.columns=['feature','score decrease']
importance_df.sort_values(by="score decrease", ascending = False)

# TECHNIQUE: Use permutation importance to identify features that can
# be dropped

# ############################### ###############################
#       Test / Train split
# ############################### ###############################
#
#
#
#
#
#
#

y = data['column_name']
X = data.drop(columns = 'column_name')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape

# ############################### ###############################
#           Imputing
# ############################### ###############################

data[['feature_a']].boxplot(); # plot before
imputer = SimpleImputer(strategy="mean")
imputer.fit(data[['feature_a']])
data['feature_a'] = imputer.transform(data[['feature_a']])
data[['feature_a']].boxplot(); # plot after


# ############################### ###############################
#           Scaling
# ############################### ###############################
#
#
#
#
#
#
#

# TODO: Cover scaler and when you use each one

# NOTES:
# Scaling is critical for many ML and DL use cases, especially those
# that are distance based

# Scaler                    Use
# ==========================================================
# StandardScaler

# MixMaxScaler              for distributions that aren't normal
#                           for ordinal (categorical with ordering) features

# RobustScaler              handles normal distributions with outliers

# Custom scaling            as per need

# StandardScaler  most simple scaling possible
std_scaler = StandardScaler()
X_scaled_train = std_scaler.fit_transform(X_train)
X_scaled_t = pd.DataFrame(X_scaled_train)

# MinMaxScale our features for you
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X.shape

# RobustScaler
r_scaler = RobustScaler() # Instanciate Robust Scaler
r_scaler.fit(data[['feature']]) # Fit scaler to feature
data['feature'] = r_scaler.transform(data[['feature']]) # apply scale

# Manual custom scaling the data:
X_unscaled = X
X_scaled = X_unscaled.copy()

for feature in X_scaled.columns:
    mu = X_scaled[feature].mean()
    sigma = X_scaled[feature].std()
    X_scaled[feature] = X_scaled[feature].apply(lambda x: (x-mu)/sigma)

X_scaled.head(3)


# ############################### ###############################
#           Feature Engineering
# ############################### ###############################
#
#
#
#
#
#
#
# TODO: go over encoding quickly

# Basic types
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# One Hot Encoding          categorical features
#                           creates a binary column for each category
# Ordinal Encoding          for
#
# Label Encoding            target only
#
# Cyclical engineering      when based on time or other cycles
#       see link
#       https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time

# One Hot Encoding
ohe = OneHotEncoder(sparse = False)
ohe.fit(data[['feature']])
feature_encoded = ohe.transform(data[['feature']])
# create new columns (tranpose), then drop original
data['feat_cat1'],data['feat_cat2'],data['feat_cat3'] = feature_encoded.T
data.drop(columns='feature', inplace=True)

# Ordinal Encoding (manual)
data['feature'] = pd.Series(np.where(data['feature']=='Y', 1, 0))
data['feature'] = np.where(data['feature'] == 'value', 1, 0)
data['feature'].value_counts()

# apply ordinal converter:
def cn_converter(x):
    if x == 'four': return 4
    if x == 'six': return 6
    if x == 'five': return 5
    if x == 'eight': return 8
    if x == 'two': return 2
    if x == 'three': return 3
    if x == 'twelve': return 12
data['feature'] = data['feature'].apply(cn_converter)

# LabelEncoder
l_encoder = LabelEncoder()
l_encoder.fit(data[['target']])
data['target'] = l_encoder.transform(data[['target']])

# Cyclical engineering (for monthly sales feature)
sns.histplot(data['MoSold']);
# split out into cos and sin to represent the cycle
data['sin_MoSold'] = np.sin(2*np.pi*data.MoSold/12)
data['cos_MoSold'] = np.cos(2*np.pi*data.MoSold/12)
# plot to ensure it worked:
data.plot.scatter('sin_MoSold','cos_MoSold').set_aspect('equal');
# then drop original feature
data.drop(columns = 'MoSold', inplace = True)

# ############################### ###############################
#           Modelling - Linear Regression
# ############################### ###############################
#
#
#
#
#
#
#

#           Modelling - Linear Regression
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# 1. basic Logistic Regression Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

log_model = LogisticRegression(max_iter = 1000)
log_model = log_model.fit(X_scaled, y)
cv_results = cross_validate(log_model, X_scaled, y, cv=5)
cv_results

# see feature names and their coefficients
lister = {}
for index, elem in enumerate(log_model.feature_names_in_):
    lister[log_model.feature_names_in_[index]] = log_model.coef_[0][index]
lister

# TODO: Classification and logistic regression: Revise log-odds


# ############################### ###############################
#           Regularisation
# ############################### ###############################

# 2.  Logistic Regression with a L2 penalty
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Let's use a **Logistic model** whose log-loss has been penalized with a
# **L2** term to figure out the **most important features** without
# overfitting.
# This is the "classification" equivalent to the "Ridge" regressor

# - By "strongly regularized" we mean "more than sklearn's default applied
#  regularization factor".
# - Default sklearn's values are very useful orders of magnitudes to keep
# in mind for "scaled features"

l1_ratio = 0 # 0 == L2 penalty
# just the l1_ratio did, but threw warnings
# set C low to emulate alpha
C = 0.001
strong_l2_log_model = LogisticRegression(C=C)
strong_l2_log_model.fit(X_scaled, y)
l2_cv_results = cross_validate(strong_l2_log_model, X_scaled, y, cv=5)
l2_cv_results

l2_lister = {}
for index, elem in enumerate(strong_l2_log_model.feature_names_in_):
    l2_lister[strong_l2_log_model.feature_names_in_[index]] = \
        strong_l2_log_model.coef_[0][index]
l2_lister

# sort and view results
df_scores = pd.DataFrame(data=strong_l2_log_model.coef_[0],
                         index=strong_l2_log_model.feature_names_in_)
abs(df_scores[0]).sort_values(ascending=True)

# 3. Logistic Regression with a L1 penalty
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# A logistic model whose log-loss has been penalized
# with a **L1** term to **filter-out the less important features**.
# This is the "classification" equivalent to the **Lasso** regressor

C=0.2
strong_l1_log_model = LogisticRegression(solver='liblinear',
                                         C=C, penalty='l1', max_iter=1000)
strong_l1_log_model.fit(X_scaled, y)
l1_cv_results = cross_validate(strong_l1_log_model, X_scaled, y, cv=5)


# ############################### ###############################
#           GridSearch, RandomSearch
# ############################### ###############################
#
#
#
#
#
#
#

# manually done first using Cross Validation
# Plot the scores as a function of K to visually find the best
# K value using the `Elbow Method`
manual = []
for k in range(1,50,1):
    knn_model = KNeighborsRegressor(n_neighbors=k)
    cv_results = cross_val_score(knn_model, X_scaled_t, y_train, cv=5)
    manual.append(cv_results.mean())

plt.plot(manual);

# GridSearch with multiple parameters
k = [1, 5, 10, 20, 50]
p = [1, 2, 3]

# Instanciate model
knn_model = KNeighborsRegressor()

# Hyperparameter Grid
grid = {
    'n_neighbors': k,
    'p': p
        }

# Instanciate Grid Search
search = GridSearchCV(
                knn_model,
                grid,
                scoring = 'r2',
                cv=5,
                n_jobs=-1
                )

# Fit data to Grid Search
search.fit(X_scaled_t, y_train)

print(f'best_score_ {search.best_score_}')
print(f'best_params_ {search.best_params_}')
print(f'best_estimator_ {search.best_estimator_}')


k = [random.randint(1,50) for x in range(20)]
p = [1, 2, 3]

# Instanciate model
knn_model = KNeighborsRegressor()

# Hyperparameter Grid
grid = {
    'n_neighbors': k,
    'p': p
        }


# Instanciate Randomised Search
search = RandomizedSearchCV(
                knn_model,
                grid,
                scoring = 'r2',
                cv=5,
                n_iter=15,
                n_jobs=-1
                )

# Fit data to Randomised Search
search.fit(X_scaled_t, y_train)

print(f'best_score_ {search.best_score_}')
print(f'best_params_ {search.best_params_}')
print(f'best_estimator_ {search.best_estimator_}')

best_p = search.best_params_['p']
best_n_neighbors = search.best_params_['n_neighbors']
best_model = KNeighborsRegressor(n_neighbors=best_n_neighbors, p=best_p)

best_model = best_model.fit(X_scaled_t, y_train)

# ############################### ###############################
#           Predict
# ############################### ###############################
#
#
#
#
#
#
#

# with best model from RandomisedSearch above
X_scaled_test = std_scaler.transform(X_test)
y_pred = best_model.predict(X_scaled_test)
r2_test = r2_score(y_test, y_pred)
r2_test
