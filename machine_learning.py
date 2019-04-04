import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, ReLU, LeakyReLU, ELU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.normalization import BatchNormalization

start_time = datetime.now()

# set random seeds for np and tf
np.random.seed(123)
tf.set_random_seed(123)

# ### READ DATA #####################################################
N_OBSER = 1  # number of observations
features_set = 3  # feature set (see prep_data_fo_nn.py)
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

# ###################################################################

# ### INPUT-OUTPUT SETTINGS, MODEL SETTINGS #########################
# 0.004 is about 1K flights
# 0.05 is about 13K flights
data_proportion = 0.05  # select only last X % of training data (e.g. 0.1 is 10%)
n_samples = int(data_proportion * X_train.shape[0])
print "Initial training size", X_train.shape[0]
X_train, y_train = X_train[-n_samples:], y_train[-n_samples:]
# X_train, X_test, y_train, y_test = X_train[:1000], X_train[-1000:], y_train[:1000], y_train[-1000:]
print "Reduced training size", X_train.shape[0]

n_features = X_train.shape[1]  # number of inputs - features
n_outputs = y_train.shape[1]  # number of outputs
n_hidden1 = 10  # number of hidden neurons in the 1st layer
n_hidden2 = 500 # number of hidden neurons in the 2nd layer
n_hidden3 = 150 # number of hidden neurons in the 3rd layer
n_hidden4 = 10  # number of hidden neurons in the 4th layer
n_hidden5 = 300  # number of hidden neurons in the 4th layer

learning_rate = 0.00001
batch_size = 1000
n_epochs = 1000000
n_layers = 1  # number of hidden layers

activation_function = 'LeakyReLU'  #  'ReLU'  'LeakyReLU' 'eLU'
if activation_function == 'LeakyReLU':
    alpha = 0.01
else:
    alpha = 0

weights_init = 'glorot_normal'   # 'he_normal' 'glorot_normal' 'glorot_uniform' 'truncated_normal'
bias_init = 'zeros'  # 'ones'

ADD_BATCH_NORM = False
USE_VAL_SET = False  # use validation set while training

checkpoint_path = "models_3/model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# ###################################################################


# ### AUXILIARY FUNCTIONS ###########################################
def custom_loss(y_true, y_pred):
    """Loss function defined as a cartesian distance in NM between true and predicted output"""
    # keras should take care of taking reduce_mean of this loss function
    # return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred, y_true)), 1)))
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred, y_true)), 1))


def dist(y_true, y_pred):
    """Metric defined as a cartesian distance in NM between true and predicted output"""
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred, y_true)), 1)))


def rmse(y_true, y_pred):
    """ Root Mean Square Error"""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def distance_np(y_true, y_pred):
    """Cartesian Distance computed using numpy.
    Gives the same result as the 'dist'

    Returns the average value"""
    return np.mean(np.sqrt((y_pred[:, 0] - y_true[:, 0])**2 + (y_pred[:, 1] - y_true[:, 1])**2))
# ###################################################################


# ### DEFINE NEURAL NETWORK #########################################
# define model as a linear stack of layers
model = Sequential()

# all model layers are defined as dense layers
# --- 1st hidden layer ----------------------------------------------
# input_shape needs to be specified for the first layer
# that takes inputs/features
model.add(Dense(units=n_hidden1,
                input_shape=(n_features,),
                use_bias=True,
                kernel_initializer=weights_init,
                bias_initializer=bias_init))

if ADD_BATCH_NORM:
    # add Batch Normalization
    model.add(BatchNormalization())

# add the activation layer explicitly
if activation_function == 'LeakyReLU':
    model.add(LeakyReLU(alpha=alpha))  # for x < 0, y = alpha*x -> non-zero slope in the negative region
elif activation_function == 'ReLU':
    model.add(ReLU())
elif activation_function == 'eLU':
    model.add(ELU())
# -------------------------------------------------------------------

if n_layers >= 2:
    print "adding 2nd hidden layer"
    # --- 2nd hidden layer ----------------------------------------------
    model.add(Dense(units=n_hidden2,
                    use_bias=True,
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init))

    if ADD_BATCH_NORM:
        model.add(BatchNormalization())

    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU(alpha=alpha))
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'eLU':
        model.add(ELU())
    # -------------------------------------------------------------------

if n_layers >= 3:
    print "adding 3rd hidden layer"
    # --- 3rd hidden layer ----------------------------------------------
    model.add(Dense(units=n_hidden3,
                    use_bias=True,
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init))

    if ADD_BATCH_NORM:
        model.add(BatchNormalization())

    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU(alpha=alpha))
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'eLU':
        model.add(ELU())
    # -------------------------------------------------------------------

if n_layers >= 4:
    print "adding 4th hidden layer"
    # --- 4th hidden layer ----------------------------------------------
    model.add(Dense(units=n_hidden4,
                    use_bias=True,
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init))

    if ADD_BATCH_NORM:
        model.add(BatchNormalization())

    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU(alpha=alpha))
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'eLU':
        model.add(ELU())
    # -------------------------------------------------------------------

if n_layers >= 5:
    print "adding 5th hidden layer"
    # --- 5th hidden layer ----------------------------------------------
    model.add(Dense(units=n_hidden4,
                    use_bias=True,
                    kernel_initializer=weights_init,
                    bias_initializer=bias_init))

    if ADD_BATCH_NORM:
        model.add(BatchNormalization())

    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU(alpha=alpha))
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'eLU':
        model.add(ELU())
    # -------------------------------------------------------------------

# --- output layer --------------------------------------------------
# no activation for the output layer
model.add(Dense(units=n_outputs,
                use_bias=True,
                kernel_initializer=weights_init,
                bias_initializer=bias_init))
# -------------------------------------------------------------------

optimizer = Adam(lr=learning_rate)

loss = MeanSquaredError()

# configure the model learning process
model.compile(optimizer=optimizer, loss=loss, metrics=[dist, rmse])

cp_callback = ModelCheckpoint(checkpoint_path,
                              verbose=1,
                              monitor='loss',
                              save_best_only=True,
                              mode='min')
# ###################################################################

# ### TRAIN NEURAL NETWORK ##########################################
# train the model, iterating the data in batches and
# displaying the training (and validation) loss and metrics for each epoch
if USE_VAL_SET:
    early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,
                        verbose=1,
                        callbacks=[cp_callback, early_stop_callback],
                        # validation_split=0.1
                        validation_data=(X_val, y_val)
                        )
else:
    early_stop_callback = EarlyStopping(monitor='loss', mode='min', patience=100, verbose=1)
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        callbacks=[early_stop_callback], epochs=n_epochs, verbose=1)
# ###################################################################

# ### PREDICTION AND EVALUATION #####################################
# compute average distance error on the training set
# and validation set
y_hat_train = model.predict(X_train, verbose=0)
y_hat_val = model.predict(X_val, verbose=0)
# y_hat_test = model.predict(X_test, verbose=0)

dist_train = distance_np(y_train, y_hat_train)
dist_val = distance_np(y_val, y_hat_val)
# dist_test = distance_np(y_test, y_hat_test)

print "average distance on the training set", np.round(dist_train, 3)
print "average distance on the validation set", np.round(dist_val, 3)
# print "average distance on the test set", np.round(dist_test, 3)

print "learning rate", learning_rate
print "batch size", batch_size
print "epochs", n_epochs
print "layers", n_layers
print "neurons in each layer", n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5
print "activation function", activation_function
print "weights initialization", weights_init
print "bias initialization", bias_init

print "training time", datetime.now() - start_time

# summarize history for loss
plt.plot(history.history['loss'])
if USE_VAL_SET:
    plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
if USE_VAL_SET:
    plt.legend(['train', 'valid'], loc='upper left')
else:
    plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for rmse
plt.plot(history.history['rmse'])
if USE_VAL_SET:
    plt.plot(history.history['val_rmse'])
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
if USE_VAL_SET:
    plt.legend(['train', 'valid'], loc='upper left')
else:
    plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for distance
plt.plot(history.history['dist'])
if USE_VAL_SET:
    plt.plot(history.history['val_dist'])
plt.title('model dist')
plt.ylabel('dist')
plt.xlabel('epoch')
if USE_VAL_SET:
    plt.legend(['train', 'valid'], loc='upper left')
else:
    plt.legend(['train'], loc='upper left')
plt.show()