## NB: this file isn't really for running
#       more for reference, pulling obvious bits together in one place

## ALL THE IMPORTS IN LWB ML AND DL

# FOR JUPYTER NOTEBOOKS ONLY:
# %load_ext autoreload
# %autoreload 2

# import fundamentals
# import preprocessing
# import scikit-learning modelling
# import model assessment
# import sklearn.metrics
# import pipelines

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import pickle

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



# Load raw data
data = pd.read_csv('path_to_csv_file.csv')


# ############################## ###############################
#       Data Exploratory Data Analysis
# ############################## ###############################

data.shape
data.head(15)
data.tail()
data.describe() # min, max, mean, count etc
data.info()

# ############################## ###############################
#       Data Visualisation
# ############################## ###############################

data.boxplot();
sns.scatterplot(data=data, x='column_name_x', y='column_name_y');
plt.plot(some_data_or_df = data);

plt.xlabel('a')
plt.ylabel('b')
plt.scatter(data['column_a'], data['column_b'], color = 'red');

# ############################### ###############################
#       Data selection
# ############################### ###############################

# Only keep numerical columns and raws without NaN
data = data.select_dtypes(include=np.number).dropna()


# ############################### ###############################
#       Test / Train split
# ############################### ###############################

y = data['column_name']
X = data.drop(columns = 'column_name')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape

# ############################### ###############################
#           Scaling
# ############################### ###############################

# most simple scaling possible
std_scaler = StandardScaler()
X_scaled_train = std_scaler.fit_transform(X_train)
X_scaled_t = pd.DataFrame(X_scaled_train)
