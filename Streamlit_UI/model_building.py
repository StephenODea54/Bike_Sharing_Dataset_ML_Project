### Imports
# Data Manipulation
import pandas as pd
import numpy as np

# Create Training Set
from sklearn.model_selection import train_test_split

# Model Training
from sklearn.ensemble import RandomForestRegressor

# To Save the model
import pickle

# Read DataFrame from GitHub Repo
url = 'https://raw.githubusercontent.com/StephenODea54/Bike_Sharing_Dataset_ML_Project/main/Data/hour.csv'
df = pd.read_csv(url)

### Dataset Transformation
# Recode Columns
df = df.astype({
    'season': 'object',
    'hr': 'object',
    'mnth': 'object',
    'holiday': 'object',
    'weekday': 'object',
    'workingday': 'object',
    'weathersit': 'object'
})

# Drop Columns
df = df.drop(columns = ['instant', 'dteday', 'yr', 'casual', 'registered', 'atemp', 'holiday', 'mnth'])

# Drop 4th Level of weathersit
df = df[df['weathersit'] != 4]

# Split Attributes and Target
# Producting testing splits seems redundant since the model will only be trained. Including it anyway to ensure that the model is the exact same
# from Sandbox.ipynb
X = df.drop('cnt', axis = 1)
y = df[['cnt']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# Instantiate Model
rf_reg = RandomForestRegressor(
    random_state = 42,
    n_estimators = 1000,
    max_features = 'log2',
    max_depth = 90
)

# Train Model
rf_reg.fit(X, y.values.ravel())

# Save Model
pickle.dump(rf_reg, open('rf_results.pkl', 'wb'))