""" Load the trained model.

NOTE! This approach does not work if the loss function in the network is defined as a class instance
(e.g. MeanSquaredError()). It results in the error:
"ValueError: Unknown entry in loss dictionary: "class_name". Only expected the following keys: [u'dense_1']"

If the loss is defined as a string value (e.g. 'mse') or a function (e.g. mse)  - than the load_model method works.
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.python.keras.models import load_model
from machine_learning import distance_np

# set random seeds for np and tf
np.random.seed(123)
tf.set_random_seed(123)

# ### READ DATA #####################################################
N_OBSER = 1  # number of observations
features_set = 2  # feature set (see prep_data_fo_nn.py)
scaling = 'min_max'  # how input is scaled
path_to_project = os.path.dirname(__file__)
path_to_prep_data = path_to_project + '/prep_data/%s/%d/%d'%(scaling, N_OBSER, features_set)

# extract training, test and validation samples
# and convert them from DataFrame to Numpy arrays
# since Keras models are trained on Numpy arrays
X_train, X_test, X_val, y_train, y_test, y_val = None, None, None, None, None, None

for data_file in os.listdir(path_to_prep_data):

    if 'input_train' in data_file:
        X_train = pd.read_pickle(path_to_prep_data + '/' + data_file).values

    elif 'input_test' in data_file:
        X_test = pd.read_pickle(path_to_prep_data + '/' + data_file).values

    elif 'input_val' in data_file:
        X_val = pd.read_pickle(path_to_prep_data + '/' + data_file).values

    elif 'output_train' in data_file:
        y_train = pd.read_pickle(path_to_prep_data + '/' + data_file).values

    elif 'output_test' in data_file:
        y_test = pd.read_pickle(path_to_prep_data + '/' + data_file).values

    elif 'output_val' in data_file:
        y_val = pd.read_pickle(path_to_prep_data + '/' + data_file).values


def dist(y_true, y_pred):
    """Metric defined as a cartesian distance in NM between true and predicted output"""
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred, y_true)), 1)))


# since the model is compiled with a custom metric 'dist'
# add it into the custom_objects attribute when loading
# the model to prevent an exception
model = load_model("models/model.h5", custom_objects={'dist': dist})

y_hat_test = model.predict(X_test, batch_size=1000, verbose=0, steps=None)

dist_test = distance_np(y_test, y_hat_test)
print "average distance on the test set", np.round(dist_test, 3)
