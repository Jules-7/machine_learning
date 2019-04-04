"""Final data preparation for a NN.

The features and labels of interest are selected from the
prepared data set. The id information (IFPLD + target_date + LAT interval)
is also selected to enable tracing back each flight for
comparing the NN performance vs the current system performance.

Scaling is applied to features to bring them on the (relatively)
same scale.

Data set is further split input/output train/test/validation sets.
After splitting, id information columns are removed and stored in
a separate file for further flights/results identification."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from CoorConv import CoorConv

# ### CHOICES TO MAKE BEFORE RUNNING THE CODE #######################
N_OBSER = 1  # 1 or 5, use this to indicate how many observations to take for training
choice = 4  # 1 or 2, choice of particular features set (see below)
scaling = 'min_max'  # 'min_max' or 'std', how to transform data
# ###################################################################

# outer keys (1, 5) are the number of observations
# inner keys are just indexes - choice (see variable above)
# the first list are all the features to extract from data
# the second list are the features to be scaled
# the last three features in the first list of features
# are not used for ML, but for identification purposes
# to be able to compare the NN predicted entries with those
# predicted by the system currently in place.
features_sets = {1: {1: [['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'DOW_SIN', 'DOW_COS', 'HOUR_SIN', 'HOUR_COS', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y']
                         ],
                     2: [['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'T1', 'DOW', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'T1', 'DOW']
                         ],
                     3: [['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y']],

                     4: [['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'NCOP_X', 'NCOP_Y']]
                     },
                 5: {1: [['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'DOW_SIN', 'DOW_COS', 'HOUR_SIN', 'HOUR_COS', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y']
                         ],
                     2: [['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'T1', 'DOW', 'IFPLID', 'TARGET_DATE', 'TO_BORDER'],
                         ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'T1', 'DOW']
                         ]}
                 }

all_features = features_sets[N_OBSER][choice][0]
scale_features = features_sets[N_OBSER][choice][1]

print 'all features\n', all_features
print 'scale features\n', scale_features

interim_data_folder = 'interim_data/5_observ'  # always read from the file containing 5 observations
prep_data_folder = 'prep_data/%s/%d/%d/' % (scaling, N_OBSER, choice)

# recursively create the folders if they don't exist
if not os.path.exists(prep_data_folder):
    os.makedirs(prep_data_folder)

data_file = os.listdir(interim_data_folder)[0]
flights_data = pd.read_pickle(interim_data_folder + '/' + data_file)
start_date, end_date = data_file.split('_')[3], data_file.split('_')[4][:-4]

# visualize correlation
# scatter_matrix(flights_data[['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'DOW_SIN', 'DOW_COS', 'HOUR_SIN', 'HOUR_COS', 'ENTRY_X', 'ENTRY_Y']])
# plt.show()

# print correlation coefficients
# if N_OBSERV == 1:
#     corr_matrix = flights_data[
#         ['X1', 'Y1', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'DOW_SIN', 'DOW_COS', 'HOUR_SIN', 'HOUR_COS', 'ENTRY_X', 'ENTRY_Y']].corr()
#
# elif N_OBSERV == 5:
#     corr_matrix = flights_data[['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5', 'NCOP_X', 'NCOP_Y', 'XCOP_X', 'XCOP_Y', 'DOW_SIN', 'DOW_COS', 'HOUR_SIN', 'HOUR_COS', 'ENTRY_X', 'ENTRY_Y']].corr()
#
# print corr_matrix[['ENTRY_X', 'ENTRY_Y']].sort_values(['ENTRY_X', 'ENTRY_Y'], ascending=[True, True])

# ToDo: remove outliers based on visualization

if 'T1' in flights_data.columns:
    # if T1 is in flights_data, perform transformation of datetime to seconds
    flights_data['T1'] = (pd.to_datetime(flights_data['T1']) - pd.to_datetime(flights_data['TARGET_DATE'])).dt.total_seconds()

# select features and labels
X = flights_data[all_features]
y = flights_data[['ENTRY_X', 'ENTRY_Y']]
print 'selected features\n', list(X)


# split data before transforming/scaling features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

X_train_std = X_train.copy()
X_test_std = X_test.copy()

# scale coordinates
if scaling == 'std':
    sc = StandardScaler()
elif scaling == 'min_max':
    sc = MinMaxScaler()

X_train_std.loc[:, scale_features] = sc.fit_transform(X_train_std[scale_features])
X_test_std.loc[:, scale_features] = sc.transform(X_test_std[scale_features])

# add validation sample
X_train_std, X_val_std, y_train, y_val = train_test_split(X_train_std, y_train, test_size=0.1, random_state=123)

print "training sample size", X_train_std.shape[0]
print "test sample size", X_test_std.shape[0]
print "validation sample size", X_val_std.shape[0]

# extract id information and store it in separate files
id_train = X_train_std[['IFPLID', 'TARGET_DATE', 'TO_BORDER']]
id_test = X_test_std[['IFPLID', 'TARGET_DATE', 'TO_BORDER']]
id_val = X_val_std[['IFPLID', 'TARGET_DATE', 'TO_BORDER']]

X_train_std.drop(columns=['IFPLID', 'TARGET_DATE', 'TO_BORDER'], inplace=True)
X_test_std.drop(columns=['IFPLID', 'TARGET_DATE', 'TO_BORDER'], inplace=True)
X_val_std.drop(columns=['IFPLID', 'TARGET_DATE', 'TO_BORDER'], inplace=True)

print "X train columns", list(X_train_std)
print "X test columns", list(X_test_std)
print "X val columns", list(X_val_std)

id_train.to_pickle(prep_data_folder + 'id_train_%s_%s.pkl' % (start_date, end_date))
id_test.to_pickle(prep_data_folder + 'id_test_%s_%s.pkl' % (start_date, end_date))
id_val.to_pickle(prep_data_folder + 'id_val_%s_%s.pkl' % (start_date, end_date))

# pickle data
X_train_std.to_pickle(prep_data_folder + 'input_train_%s_%s.pkl' % (start_date, end_date))
X_test_std.to_pickle(prep_data_folder + 'input_test_%s_%s.pkl' % (start_date, end_date))
X_val_std.to_pickle(prep_data_folder + 'input_val_%s_%s.pkl' % (start_date, end_date))

y_train.to_pickle(prep_data_folder + 'output_train_%s_%s.pkl' % (start_date, end_date))
y_test.to_pickle(prep_data_folder + 'output_test_%s_%s.pkl' % (start_date, end_date))
y_val.to_pickle(prep_data_folder + 'output_val_%s_%s.pkl' % (start_date, end_date))
