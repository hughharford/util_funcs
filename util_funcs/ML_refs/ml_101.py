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
from sklearn.compose import make_column_selector
from sklearn.compose import make_pipeline
from sklearn.pipeline import make_union
from sklearn.pipeline import make_column_transformer

from sklearn.metrics import make_scorer, mean_squared_log_error
# import graphviz
from sklearn.tree import export_graphviz
from xgboost import XGBRegressor

# get diagram with pipeline name
from sklearn import set_config; # set_config(display='diagram')

# import selected parts of tensorflow tf
import tensorflow as tf
# tensorflow yellow underlines don't stop it working...
from tensorflow.keras import dfsets
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, SimpleRNN, LSTM, GRU
from tensorflow.keras.metrics import Accuracy, Precision

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from tensorflow.keras.preprocessing import image_dfset_from_directory

# padding with tf / keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking

# ############################## ###############################
#       Loading df
# ############################## ###############################
#
#
#
#
#
#
#

df = pd.read_csv('path_to_csv_file.csv')


# ############################## ###############################
#       Exploratory df Analysis and df Cleaning
# ############################## ###############################
#
#
#
#
#
#
#

df.shape
df.head(15)
df.tail()
df.describe() # min, max, mean, count etc
df.info()

df.columns # see column names

df[['feature']].value_counts()
df['feature'].value_counts().sort_values()

df.Feature.unique() # to show uniques


# Query the df:
#Using variable
value=False
df.query("feature == @value")
df.query("`feature1` >= @value*2 and `feature2` <= @value*3")

# ############################## ###############################
#       df Cleaning
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
duplicate_count = df.duplicated().sum()
duplicate_count

# drop all duplicates
df = df.drop_duplicates()


#       Missing df
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# list empty values as a percentage by feature
empty = df.isnull().sum().sort_values(ascending=False);
empty / len(df)

#       Replacing values
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
df.FeatureName.replace(np.nan, "NoG", inplace=True) #Replace NaN with...
df[['feature']] = df[['feature']].replace(
    np.nan, df['feature'].median()) # replace with median


# ############################## ###############################
#       df Visualisation
# ############################## ###############################
#
#
#
#
#
#
#

# simple plots
plt.plot(some_df = df);

# chart features
plt.xlabel('a')
plt.ylabel('b')


# boxplot
df.boxplot(); # good for seeing outliers

# histograms
df.hist('feature_a'); # pd histogram
sns.histplot(df=df,x="feature",bins=30);
sns.distplot(x=df['feature'], kde = False);

# scatter plots
sns.scatterplot(df=df, x='column_name_x', y='column_name_y');
plt.scatter(df['column_a'], df['column_b'], color = 'red');
# for showing cyclical values
df.plot.scatter('sin_MoSold','cos_MoSold').set_aspect('equal');

# very cool 3d, interactive plot
import plotly.graph_objects as go
# for use in a Jupyter notebook...?
Z, range_a, range_b = 50,50,50
surface = go.Surface(x=range_a, y=range_b, z=Z)
scatter = go.Scatter3d(x=df['column_a'], y=df['column_b'],
                       z=df['loss_history'], mode='markers')
fig = go.Figure(df=[surface, scatter])

fig.update_layout(title='Loss Function', autosize=False,
                  width=1000, height=800)
fig.show()

# correlation heatmap
corr = df.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= "YlGnBu");
# show the correlation between column pairs in a dfframe.
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
df = df.drop(columns = 'feature')

# Only keep numerical columns and raws without NaN
df = df.select_dtypes(include=np.number).dropna()

X_num = X.select_dtypes(include=np.number)
X_cat = X.select_dtypes(exclude=np.number)

# select by column name
df = df[['column_one','column_two']]

y = df['column_name']
X = df.drop(columns = 'column_name')

# correlation heatmap
corr = df.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap= "YlGnBu");
# show the correlation between column pairs in a dfframe.
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
importance_df = pd.dfFrame(np.vstack((X.columns,permutation_score.importances_mean)).T)
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

y = df['column_name']
X = df.drop(columns = 'column_name')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape

# ############################### ###############################
#       Pipelines
# ############################### ###############################
#
#
#
#
#
#
#

# simple pipeline with model:
pipe = Pipeline([
    ('KNNImputer', KNNImputer())
    ,('MixMaxScaler', MinMaxScaler())
    ,('LogisticRegression', LogisticRegression())
])
# then fit and score (runs last section's methods, ie. the model):
pipe.fit(X_train, y_train)
pipe.score(X, y)

pipe # in JN, will show the pipe


# Impute then scale numerical values:
num_transformer = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

# Encode categorical values
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Parallelize "num_transformer" and "cat_transfomer"
preprocessor = ColumnTransformer([
    ('num_tr', num_transformer, ['age', 'bmi']),
    ('cat_tr', cat_transformer, ['smoker', 'region'])
])

# custom functions for pipelines must have .fit() and .transform():

# TransformerMixin inheritance is used to create fit_transform() method from fit() and transform()
from sklearn.base import TransformerMixin, BaseEstimator

class CustomStandardizer(TransformerMixin, BaseEstimator):

    def __init__(self):
        # super.__init__(self)
        pass

    def fit(self, X, y=None):
        self.x_mean = X.mean()
        self.x_sigma = X.std(ddof=0)
        # print(f'self.x_mean \n{self.x_mean}\n')
        # print(f'self.x_sigma \n{self.x_sigma}\n')
        return self

    def transform(self, X, y=None):
        return (X - self.x_mean) /  self.x_sigma


num_col = make_column_selector(dtype_include=['float64'])

cat_transformer = OneHotEncoder()
cat_col = make_column_selector(dtype_include=['object','bool'])


# these column transforms are equivalent:

#1:
# Parallelize "num_transformer" and "cat_transfomer"
# preprocessor = ColumnTransformer([
#     ('num_tr', num_transformer, X_num),
#     ('cat_tr', cat_transformer, X_cat),
#     remainder='passthrough'
#     ])
#2:
preprocess_columns = make_column_transformer(
    (num_transformer, num_col),
    (cat_transformer, cat_col),
    remainder='passthrough'
)

# finally, add a model:
# Add estimator
pipe = make_pipeline(preprocessor, Ridge())
pipe

# ############################### ###############################
#           Imputing
# ############################### ###############################

df[['feature_a']].boxplot(); # plot before
imputer = SimpleImputer(strategy="mean")
imputer.fit(df[['feature_a']])
df['feature_a'] = imputer.transform(df[['feature_a']])
df[['feature_a']].boxplot(); # plot after


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
X_scaled_t = pd.dfFrame(X_scaled_train)

# MinMaxScale our features for you
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.dfFrame(X_scaled, columns=X.columns)
X.shape

# RobustScaler
r_scaler = RobustScaler() # Instanciate Robust Scaler
r_scaler.fit(df[['feature']]) # Fit scaler to feature
df['feature'] = r_scaler.transform(df[['feature']]) # apply scale

# Manual custom scaling the df:
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
ohe.fit(df[['feature']])
feature_encoded = ohe.transform(df[['feature']])
# create new columns (tranpose), then drop original
df['feat_cat1'],df['feat_cat2'],df['feat_cat3'] = feature_encoded.T
df.drop(columns='feature', inplace=True)

# Ordinal Encoding (manual)
df['feature'] = pd.Series(np.where(df['feature']=='Y', 1, 0))
df['feature'] = np.where(df['feature'] == 'value', 1, 0)
df['feature'].value_counts()

# apply ordinal converter:
def cn_converter(x):
    if x == 'four': return 4
    if x == 'six': return 6
    if x == 'five': return 5
    if x == 'eight': return 8
    if x == 'two': return 2
    if x == 'three': return 3
    if x == 'twelve': return 12
df['feature'] = df['feature'].apply(cn_converter)

# LabelEncoder
l_encoder = LabelEncoder()
l_encoder.fit(df[['target']])
df['target'] = l_encoder.transform(df[['target']])

# Cyclical engineering (for monthly sales feature)
sns.histplot(df['MoSold']);
# split out into cos and sin to represent the cycle
df['sin_MoSold'] = np.sin(2*np.pi*df.MoSold/12)
df['cos_MoSold'] = np.cos(2*np.pi*df.MoSold/12)
# plot to ensure it worked:
df.plot.scatter('sin_MoSold','cos_MoSold').set_aspect('equal');
# then drop original feature
df.drop(columns = 'MoSold', inplace = True)

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
df_scores = pd.dfFrame(df=strong_l2_log_model.coef_[0],
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

# Fit df to Grid Search
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

# Fit df to Randomised Search
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
