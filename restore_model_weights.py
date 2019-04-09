import os
import numpy as np
import tensorflow as tf
import pandas as pd

from machine_learning import distance_np
from machine_learning import FeedForwardNeuralNet
from tensorflow.python.keras.losses import mse

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

n_features = X_train.shape[1]  # number of inputs - features
n_outputs = y_train.shape[1]  # number of outputs
# number of hidden neurons per layer
n_hidden = {1: 10,
            2: 5,
            3: 100,
            4: 10,
            5: 300}

learning_rate = 0.00001
n_layers = 1  # number of hidden layers

# restore model using weights only
fnn = FeedForwardNeuralNet(X_train=X_train, X_test=X_test, X_val=None,
                           y_train=y_train, y_test=y_test, y_val=None,
                           n_features=n_features, n_outputs=n_outputs,
                           n_layers=n_layers, n_hidden=n_hidden,
                           lr=learning_rate, batch=1000,
                           epochs=100, activation='LeakyReLU',
                           alpha=0.01, weights='glorot_normal',
                           bias='zeros', loss=mse)

fnn.create_model()
fnn.model.load_weights("weights/model_weights.h5")

print 'test size', X_test.shape[0]

y_hat_test = fnn.model.predict(X_test, batch_size=1000, verbose=0, steps=None)
dist_test = distance_np(y_test, y_hat_test)
print "average distance on the test set", np.round(dist_test, 3)


