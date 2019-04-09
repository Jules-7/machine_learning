import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, ReLU, LeakyReLU, ELU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import mse
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.normalization import BatchNormalization

# set random seeds for np and tf
np.random.seed(123)
tf.set_random_seed(123)


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


class FeedForwardNeuralNet(object):

    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val,
                 n_features, n_outputs, n_layers, n_hidden, lr, batch, epochs,
                 activation, alpha=None, weights='glorot_normal',
                 bias='zeros', loss='mse', batch_norm=False, cp_path=None, cp_dir=None):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_layers = n_layers  # number of hidden layers
        self.n_hidden = n_hidden  # neurons per each hidden layer
        self.lr = lr  # learning rate
        self.batch_size = batch
        self.epochs = epochs
        self.activation = activation
        self.alpha = alpha  # used for LeakyReLU activation
        self.weights_init = weights
        self.bias_init = bias
        self.loss = loss
        self.batch_norm = batch_norm
        self.cp_path = cp_path  # checkpoint path
        self.cp_dir = cp_dir  # checkpoint dir

    def create_model(self):
        """ DEFINE NEURAL NETWORK """
        # define model as a linear stack of dense layers
        self.model = Sequential()

        # iteratively add hidden layers
        for layer_n in range(1, self.n_layers+1):
            print layer_n, "hidden layer\n",
            if layer_n == 1:  # input_shape needs to be specified for the first layer
                self.model.add(Dense(units=self.n_hidden[layer_n], input_shape=(self.n_features,),
                                     kernel_initializer=self.weights_init, bias_initializer=self.bias_init))
            else:
                self.model.add(Dense(units=self.n_hidden[layer_n], kernel_initializer=self.weights_init,
                                     bias_initializer=self.bias_init))

            if self.batch_norm:
                self.model.add(BatchNormalization())  # add batch normalization before activation

            # add the activation layer explicitly
            if self.activation == 'LeakyReLU':
                self.model.add(LeakyReLU(alpha=self.alpha))  # for x < 0, y = alpha*x -> non-zero slope in the negative region

            elif self.activation == 'ReLU':
                self.model.add(ReLU())

            elif self.activation == 'eLU':
                self.model.add(ELU())

        # add output layer; no activation for the output layer
        self.model.add(Dense(units=self.n_outputs, kernel_initializer=self.weights_init,
                             bias_initializer=self.bias_init))

    def compile_model(self):

        optimizer = Adam(lr=self.lr)  # hardcoded Adam optimizer

        # configure the model learning process
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=[dist])

    def set_callback(self):
        # checkpoint to store model weights
        self.checkpoint = ModelCheckpoint(self.cp_path, verbose=1, monitor='loss',
                                          save_weights_only=True, mode='min')

        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)

    def train_model(self):
        # train the model iterating data in batches and
        # storing the training and validation loss and metrics for each epoch
        start_time = datetime.now()
        if self.X_val:
            self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, verbose=1,
                                     callbacks=[self.checkpoint, self.early_stop],
                                     validation_data=(self.X_val, self.y_val))
        else:
            self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, verbose=1,
                                     callbacks=[self.checkpoint, self.early_stop],
                                     validation_split=0.1)
        print "training time", datetime.now() - start_time, '\n'

    def predict(self):
        # compute average distance error on the training and validation set
        y_hat_train = self.model.predict(self.X_train, verbose=0)
        # y_hat_test = self.model.predict(self.X_test, verbose=0)

        dist_train = distance_np(y_train, y_hat_train)
        # dist_test = distance_np(y_test, y_hat_test)

        print "average distance on the training set", np.round(dist_train, 3)
        # print "average distance on the test set", np.round(dist_test, 3)

        if self.X_val:
            y_hat_val = self.model.predict(self.X_val, verbose=0)
            dist_val = distance_np(y_val, y_hat_val)
            print "average distance on the validation set", np.round(dist_val, 3)

    def model_structure(self):
        print "\nPrinting model structure:\n"
        print "learning rate\t", self.lr
        print "batch size\t\t", self.batch_size
        print "epochs\t\t\t\t", self.epochs
        print "layers\t\t\t", self.n_layers
        print "neurons in each layer\t", self.n_hidden
        print "activation function\t", self.activation
        print "weights initialization\t", self.weights_init
        print "bias initialization\t", self.bias_init
        print "batch normalization\t", self.batch_norm
        print N_OBSER, "observ, feature set", features_set,

    def summarize_history(self):
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(['train', 'valid'], loc='upper left')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

        # # summarize history for rmse
        # plt.plot(history.history['rmse'])
        # if USE_VAL_SET:
        #     plt.plot(history.history['val_rmse'])
        # plt.title('model rmse')
        # plt.ylabel('rmse')
        # plt.xlabel('epoch')
        # if USE_VAL_SET:
        #     plt.legend(['train', 'valid'], loc='upper left')
        # else:
        #     plt.legend(['train'], loc='upper left')
        # plt.show()
        #
        # summarize history for distance
        plt.plot(self.history.history['dist'])
        plt.plot(self.history.history['val_dist'])
        plt.title('model dist')
        plt.ylabel('dist')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()


if __name__ == "__main__":

    # ### READ DATA #####################################################
    N_OBSER = 1  # number of observations
    features_set = 2  # feature set (see prep_data_fo_nn.py)
    scaling = 'min_max'  # how input is scaled
    path_to_project = os.path.dirname(__file__)
    path_to_prep_data = path_to_project + '/prep_data/%s/%d/%d' % (scaling, N_OBSER, features_set)

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
    data_proportion = 1  # select only last X % of training data (e.g. 0.1 is 10%)
    n_samples = int(data_proportion * X_train.shape[0])
    print "Initial training size", X_train.shape[0]
    X_train, y_train = X_train[-n_samples:], y_train[-n_samples:]
    # X_train, X_test, y_train, y_test = X_train[:1000], X_train[-1000:], y_train[:1000], y_train[-1000:]
    print "Reduced training size", X_train.shape[0]

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

    #weights_init options # 'he_normal'  # 'glorot_normal'  'glorot_uniform' 'truncated_normal'

    checkpoint_path = "weights/model_weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # ###################################################################

    fnn = FeedForwardNeuralNet(X_train=X_train, X_test=None, X_val=None,
                               y_train=y_train, y_test=None, y_val=None,
                               n_features=n_features, n_outputs=n_outputs,
                               n_layers=n_layers, n_hidden=n_hidden,
                               lr=learning_rate, batch=1000,
                               epochs=100, activation='LeakyReLU',
                               alpha=0.01, weights='glorot_normal',
                               bias='zeros', loss=mse, batch_norm=False,
                               cp_path=checkpoint_path, cp_dir=checkpoint_dir)

    fnn.create_model()
    fnn.compile_model()
    fnn.set_callback()
    fnn.train_model()
    fnn.model.save('models/model.h5')  # save complete model
    fnn.predict()
    fnn.summarize_history()
    fnn.model_structure()


