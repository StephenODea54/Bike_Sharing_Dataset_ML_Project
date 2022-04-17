### Imports
# Data Manipulation
import pandas as pd
import numpy as np

# Create Training Set
from sklearn.model_selection import train_test_split

# Model Training
from sklearn.ensemble import GradientBoostingRegressor

# To Save the model
import joblib

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
gb_reg = GradientBoostingRegressor(
    n_estimators = 1000,
    min_samples_leaf = 5,
    max_features = 0.3,
    max_depth = 10,
    learning_rate = 0.01,
    random_state = 42
)

# Train Model
gb_reg.fit(X, y.values.ravel())

# Save Model
joblib.dump(gb_reg, 'gb_results.pkl', compress = 3)