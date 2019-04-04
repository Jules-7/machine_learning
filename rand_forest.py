import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# set random seeds for np
np.random.seed(123)


N_OBSER = 1  # number of observations
features_set = 4  # feature set (see prep_data_fo_nn.py)
scaling = 'min_max'  # how input is scaled
path_to_project = os.path.dirname(__file__)
path_to_prep_data = path_to_project + '/prep_data/%s/%d/%d'%(scaling, N_OBSER, features_set)

# extract training and test samples
X_train, X_test, y_train, y_test = None, None, None, None

for data_file in os.listdir(path_to_prep_data):
    if 'input_train' in data_file:
        X_train = pd.read_pickle(path_to_prep_data + '/' + data_file)
    elif 'input_test' in data_file:
        X_test = pd.read_pickle(path_to_prep_data + '/' + data_file)
    elif 'output_train' in data_file:
        y_train = pd.read_pickle(path_to_prep_data + '/' + data_file)
    elif 'output_test' in data_file:
        y_test = pd.read_pickle(path_to_prep_data + '/' + data_file)

data_proportion = 1.0  # select only last X % of training data (e.g. 0.1 is 10%)
n_samples = int(data_proportion * X_train.shape[0])
print "Initial training size", X_train.shape[0]
X_train, y_train = X_train[-n_samples:], y_train[-n_samples:]
print "Reduced training size", X_train.shape[0]

features_names = list(X_train)

X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values

model = RandomForestRegressor(n_estimators=10,
                              random_state=123,
                              # max_depth=100,
                              # min_samples_split=50,
                              verbose=1,
                              n_jobs=-1)

print "fitting model"
model.fit(X_train, y_train)

print "feature importance\n"
for name, score in zip(features_names, model.feature_importances_):
    print "%s \t %.6f"%(name, score)

# ###################################################################
print "predicting training output"
y_hat = model.predict(X_train)

print "predicting test output"
y_hat_test = model.predict(X_test)

# ###################################################################


def distance_np(y_true, y_pred):
    """Cartesian Distance computed using numpy.
    Gives the same result as the 'dist'

    Returns the average value"""
    return np.mean(np.sqrt((y_pred[:, 0] - y_true[:, 0])**2 + (y_pred[:, 1] - y_true[:, 1])**2))


print "training dist", distance_np(y_train, y_hat)
print "test dist", distance_np(y_test, y_hat_test)

mse_train = mean_squared_error(y_train, y_hat)
rmse_train = np.sqrt(mse_train)
print "training rmse", rmse_train

mse_test = mean_squared_error(y_test, y_hat_test)
rmse_test = np.sqrt(mse_test)
print "test rmse", rmse_test
