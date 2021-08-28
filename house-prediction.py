# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data, and separate the target
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

features = [
    'MSSubClass',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
]
# Select columns corresponding to features, and preview the data
X = home_data[features]
# X = X.fillna(-1)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1, n_estimators=700)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)

# Defining Gradient boosting regressor
gbr_model = GradientBoostingRegressor(random_state=1, n_estimators=500)
gbr_model.fit(train_X, train_y)
gbr_val_predictions = gbr_model.predict(val_X)

final_prediction = (rf_val_predictions + gbr_val_predictions) /2


rf_val_rmse = np.sqrt(mean_squared_error(final_prediction, val_y))
print("Validation RMSE for Random Forest Model: {:,.0f}".format(rf_val_rmse))

# gbr_val_rmse = np.sqrt(mean_squared_error(gbr_val_predictions, val_y))
# print("Validation MAE for Gradient: {:,.0f}".format(gbr_val_rmse))


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model = RandomForestRegressor(random_state=1, n_estimators=700)

# Defining Gradient boosting regressor
gbr_model = GradientBoostingRegressor(random_state=1, n_estimators=500)

final_prediction = (rf_val_predictions + gbr_val_predictions) /2

# fit model on all data from the training data
rf_model.fit(X, y)
gbr_model.fit(X, y)


# path to file you will use for predictions
test_data_path = '../input/home-data-for-ml-course/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_data = test_data.fillna(-1)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
rf_val_predictions = rf_model.predict(test_X)
gbr_val_predictions = gbr_model.predict(test_X)

test_preds = (rf_val_predictions + gbr_val_predictions)/2


# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
