# Script dependencies
# Libraries for Anaysis
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import math

# Libraries for Plotting Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Libraries for data preparation and model building
from sklearn.model_selection import train_test_split    # To split the data into training and testing data
from sklearn.preprocessing import StandardScaler        # For standardizing features
from sklearn.linear_model import LinearRegression       # For the LINEAR Model from Sklearn
from sklearn.linear_model import Ridge                  # For the RIDGE Regression module from sklearn
from sklearn.linear_model import Lasso                  # For the LASSO Model from Sklearn
from sklearn.model_selection import GridSearchCV        # To sort out our Hyper_Parameters
from sklearn.tree import DecisionTreeRegressor          # For the Decision-Tree Model
from sklearn.ensemble import RandomForestRegressor      # For the RandomForest Model
import xgboost as xgb                                   # For the xgBoost Model
from sklearn.svm import SVR

# Libraries for calculating performance metrics
from sklearn.metrics import mean_squared_error          # Apply np.sqrt MSE to get RMSE
import time                                             # For Calulating ALgo Time Run

# Libraries to Save/Restore Models
import pickle

# Mute warnings
import warnings
warnings.filterwarnings('ignore')



# We make use of an xgboost model trained on .
model=pickle.load(open('resources/models/220729_xgb_.pkl', 'rb'))


# Importing data
df = pd.read_csv("resources/datasets/Our_CO2emission_Modelling_Data.csv")
# Drop Unamed Column & predict target
df = df.drop(['Unnamed: 0',"CO2_emission"], axis=1)

print(df.columns)

# Extracting features and label; Readying for Split 
X = df.drop(['CO2_emission'], axis=1)
y = df['CO2_emission']

# Standardize/Scale our Dataset
scaler = StandardScaler() # create scaler object

# convert the scaled predictor values into a dataframe
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
    #        Modeling
# -------------------------------------------------------------------
def xgboost_model():
    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, shuffle=False, random_state=42)

    #create Xgboost
    xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.6, learning_rate = 0.1, max_depth = 5, alpha = 6, n_estimators = 100, subsample = 0.7)  

    # Fit the Ml model
    # XGBoost gbtree
    xgb_reg.fit(x_train, y_train)

    # evaluate xgboost model [Getting Predictions]
    y_pred_xgb = xgb_reg.predict(x_test)

    # ---------------------------------------------------------
    ### Remodelling with Essential Features ###
    # ---------------------------------------------------------

    # Drop Non-Essentials
    new_df = df.drop(['Agric_GDP', 'ei_gdp', 'pop_growth', 'pop_density', 'Deforestation', 'Population'], axis=1)

    # Extracting features and label; Readying for Split 
    X_new = new_df.drop(['CO2_emission'], axis=1)
    y_new = new_df['CO2_emission']

    # convert the scaled predictor values into a dataframe
    X_scaled_new = pd.DataFrame(scaler.fit_transform(X_new), columns=X_new.columns)

    # splitting data
    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_scaled_new, y_new, test_size=0.15, shuffle=False, random_state=42)

    # Create Xgboost Model
    xgb_reg_new = xgb.XGBRegressor(objective='count:poisson', colsample_bytree=0.6, learning_rate=0.1, max_depth=3, alpha=6, n_estimators=600, subsample=0.7) #'reg:squarederror'

    # Train Tree Model
    xgb_reg_new.fit(x_train_new, y_train_new)

    return 


    """load model
    select rand clumn in data
    prepare data to predict ie.
    drop unneeded columns
    scaling

    """





