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
#       Data Exploratory Data Analysis
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

data.boxplot();
sns.scatterplot(data=data, x='column_name_x', y='column_name_y');
plt.plot(some_data_or_df = data);

plt.xlabel('a')
plt.ylabel('b')
plt.scatter(data['column_a'], data['column_b'], color = 'red');

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

# ############################### ###############################
#       Data selection
# ############################### ###############################
#
#
#
#
#
#
#

# Only keep numerical columns and raws without NaN
data = data.select_dtypes(include=np.number).dropna()


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
#           Scaling
# ############################### ###############################
#
#
#
#
#
#
#

# most simple scaling possible
std_scaler = StandardScaler()
X_scaled_train = std_scaler.fit_transform(X_train)
X_scaled_t = pd.DataFrame(X_scaled_train)

# MinMaxScale our features for you
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X.shape

# ############################### ###############################
#           Regularisation
# ############################### ###############################
#
#
#
#
#
#
#




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
    l2_lister[strong_l2_log_model.feature_names_in_[index]] = strong_l2_log_model.coef_[0][index]
l2_lister

# sort results
df_scores = pd.DataFrame(data=strong_l2_log_model.coef_[0], index=strong_l2_log_model.feature_names_in_)
abs(df_scores[0]).sort_values(ascending=True)
