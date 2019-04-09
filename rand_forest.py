import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score


# set random seeds for np
np.random.seed(123)


N_OBSER = 1  # number of observations
features_set = 1  # feature set (see prep_data_fo_nn.py)
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

# ###################################################################
# test this code
# gsc = GridSearchCV(estimator=RandomForestRegressor(),
#                    param_grid={
#                        'max_depth': range(10, 100),
#                        'n_estimators': (10, 50, 100, 1000),
#                    },
#                    cv=5,
#                    scoring='neg_mean_squared_error',
#                    verbose=0,
#                    n_jobs=-1)
#
# print "fitting model"
# grid_result = gsc.fit(X_train, y_train)
# best_params = grid_result.best_params_
#
# rfr = RandomForestRegressor(max_depth=best_params["max_depth"],
#                             n_estimators=best_params["n_estimators"],
#                             random_state=False,
#                             verbose=False)
#
# scores = cross_val_score(rfr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
#
# predictions = cross_val_predict(rfr, X, y, cv=10)
# ###################################################################


model = RandomForestRegressor(n_estimators=50,  # number of trees
                              random_state=123,
                              max_depth=100,  # tree depth - number of splits
                              min_samples_leaf=2,
                              max_features=3,
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
