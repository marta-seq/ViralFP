import os
import pandas as pd
import numpy as np
from tensorflow import keras
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf


import os
import pandas as pd
import numpy as np
from tensorflow import keras
import random
import tensorflow as tf

from tensorflow.keras import optimizers
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers, Input
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
# from nlf_blosum_encoding import blosum_encode
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GRU, BatchNormalization, Masking

from propythia_src.deep_ml import DeepML

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(tf.executing_eagerly())
print('execution eager')

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

def lstm_simple(input_dim, number_classes, timestep = 21,
                  optimizer='Adam',
                  lstm_layers=(64, 64, 32),
                  bidirectional = True,
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  dropout_rate=(0.1, 0.1, 0.1),
                  l1=1e-5, l2=1e-4,
                  dense_layers=(32, 16),
                  dropout_rate_dense=(0.1, 0.1),
                  dense_activation="relu", loss='sparse_categorical_crossentropy',final_activation='softmax', metric ='accuracy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, timestep,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        if bidirectional:
            for layer in range(len(lstm_layers) - 1):
                    model.add(Bidirectional(
                        LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                             recurrent_activation=recurrent_activation,
                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                             dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 21,)))

            # add last lstm1 layer
            model.add(Bidirectional(
                LSTM(units=lstm_layers[-1], return_sequences=False,
                     activation=activation, recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[-1], recurrent_dropout=0.0)))
        else:
            for layer in range(len(lstm_layers) - 1):
                model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                               recurrent_activation=recurrent_activation,
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                               dropout=dropout_rate[layer], recurrent_dropout=0.0))

            # add last lstm1 layer
            model.add(
                LSTM(units=lstm_layers[-1], return_sequences=False,
                     activation=activation, recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[-1], recurrent_dropout=0.0))
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=final_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        print(model.summary())
        return model


def lstm_embedding(input_dim, number_classes, timestep = 21,output_dim=21,
                  optimizer='Adam',
                  lstm_layers=(64, 64, 32),
                  bidirectional = True,
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  dropout_rate=(0.1, 0.1, 0.1),
                  l1=1e-5, l2=1e-4,
                  dense_layers=(32, 16),
                  dropout_rate_dense=(0.1, 0.1),
                  dense_activation="relu", loss='sparse_categorical_crossentropy',final_activation='softmax'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim,), name='main_input'))
        # add initial dropout
        model.add(Masking(mask_value=0))
        model.add(Embedding(input_dim=timestep, output_dim=output_dim))
        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        if bidirectional:
            for layer in range(len(lstm_layers) - 1):
                    model.add(Bidirectional(
                        LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                             recurrent_activation=recurrent_activation,
                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                             dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, output_dim,)))
            # add last lstm1 layer
            model.add(Bidirectional(
                LSTM(units=lstm_layers[-1], return_sequences=False,
                     activation=activation, recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[-1], recurrent_dropout=0.0)))
        else:
            for layer in range(len(lstm_layers) - 1):
                model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                               recurrent_activation=recurrent_activation,
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                               dropout=dropout_rate[layer], recurrent_dropout=0.0))

            # add last lstm1 layer
            model.add(
                LSTM(units=lstm_layers[-1], return_sequences=False,
                     activation=activation, recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[-1], recurrent_dropout=0.0))
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=final_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model



class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


def bilstm_attention(input_dim, number_classes,
                     n_features=20,
                     optimizer='Adam',
                     lstm_layers=(64, 64, 32),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout_rate=(0.1, 0.1, 0.1),
                     l1=1e-5, l2=1e-4,
                     dense_layers=(32, 16),
                     dropout_rate_dense=(0.1, 0.1),
                     dense_activation="relu", loss='sparse_categorical_crossentropy', final_activation='softmax', metric = 'accuracy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, n_features,), dtype='float32', name='main_input'))

        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)

        # receives LSTM with return sequences =True

        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation=final_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric]) # metrics accuracy  mse
        print(model.summary())
        return model


def dnn_simple(input_dim, number_classes,
               hidden_layers=(128, 64, 32),
               optimizer='Adam',
               initial_dropout_value=0.0,
               dropout_rate=(0.3,),
               batchnormalization=(True,), activation="relu",
               l1=1e-5, l2=1e-4, loss_fun='binary_crossentropy', activation_fun='sigmoid'):

    last_layer = hidden_layers[-1]
    if len([dropout_rate]) == 1:
        dropout_rate = list(dropout_rate * len(hidden_layers))
    if len([batchnormalization]) == 1:
        batchnormalization = list(batchnormalization * len(hidden_layers))

    # with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    if initial_dropout_value > 0:
        model.add(Dropout(initial_dropout_value))

    for layer in range(len(hidden_layers)):
        model.add(Dense(units=hidden_layers[layer], activation=activation,
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        if batchnormalization[layer]:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate[layer]))

    # # last layer
    # model.add(Dense(units=last_layer, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # if batchnormalization: model.add(BatchNormalization())
    # if final_dropout_value > 0: model.add(Dropout(final_dropout_value))

    # Add Classification Dense, Compile model and make it ready for optimization
    model.add(Dense(1, activation=activation_fun))
    model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])

    return model

# def gru_simple(input_dim, number_classes,
#                optimizer='Adam',
#                lstm_layers=(64, 64, 32),
#                activation='tanh',
#                recurrent_activation='sigmoid',
#                dropout_rate=(0.1, 0.1, 0.1),
#                l1=1e-5, l2=1e-4,
#                dense_layers=(32, 16),
#                dropout_rate_dense=(0.1, 0.1),
#                dense_activation="relu", loss='sparse_categorical_crossentropy'):
#     with strategy.scope():
#         model = Sequential()
#         # input dim timesteps = seq size , features. 21 features per character
#         model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
#         # add initial dropout
#
#         # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
#         model.add(Masking(mask_value=0))
#         for layer in range(len(lstm_layers) - 1):
#             model.add(GRU(units=lstm_layers[layer], return_sequences=True, activation=activation,
#                           recurrent_activation=recurrent_activation,
#                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                           dropout=dropout_rate[layer], recurrent_dropout=0.0))
#
#         # add last lstm1 layer
#         model.add(
#             GRU(units=lstm_layers[-1], return_sequences=False,
#                 activation=activation, recurrent_activation=recurrent_activation,
#                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                 dropout=dropout_rate[-1], recurrent_dropout=0.0))
#
#         # add denses
#         for layer in range(len(dense_layers)):
#             model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate_dense[layer]))
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(number_classes, activation='softmax'))
#         model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#         print(model.summary())
#         return model
#
#
#
# def lstm_attention(input_dim, number_classes,
#                    optimizer='Adam',
#                    lstm_layers=(64, 64, 32),
#                    activation='tanh',
#                    recurrent_activation='sigmoid',
#                    dropout_rate=(0.1, 0.1, 0.1),
#                    l1=1e-5, l2=1e-4,
#                    dense_layers=(32, 16),
#                    dropout_rate_dense=(0.1, 0.1),
#                    dense_activation="relu", loss='sparse_categorical_crossentropy'):
#     with strategy.scope():
#         model = Sequential()
#         # input dim timesteps = seq size , features. 21 features per character
#         model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
#         # add initial dropout
#
#         # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
#         model.add(Masking(mask_value=0))
#         for layer in range(len(lstm_layers)):
#             model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
#                            recurrent_activation=recurrent_activation,
#                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                            dropout=dropout_rate[layer], recurrent_dropout=0.0))
#
#         # receives LSTM with return sequences =True
#
#         # add attention
#         # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
#         model.add(attention())
#         # a, context = attention()(model)
#         # add denses
#         for layer in range(len(dense_layers)):
#             model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate_dense[layer]))
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(number_classes, activation='softmax'))
#         model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#         print(model.summary())
#
#         return model
#
#
# def gru_attention(input_dim, number_classes,
#                   optimizer='Adam',
#                   lstm_layers=(64, 64, 32),
#                   activation='tanh',
#                   recurrent_activation='sigmoid',
#                   dropout_rate=(0.1, 0.1, 0.1),
#                   l1=1e-5, l2=1e-4,
#                   dense_layers=(32, 16),
#                   dropout_rate_dense=(0.1, 0.1),
#                   dense_activation="relu", loss='sparse_categorical_crossentropy'):
#     with strategy.scope():
#         model = Sequential()
#         # input dim timesteps = seq size , features. 21 features per character
#         model.add(Input(shape=(input_dim, 21,), dtype='float32', name='main_input'))
#         # add initial dropout
#
#         # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
#         model.add(Masking(mask_value=0))
#         for layer in range(len(lstm_layers)):
#             model.add(Bidirectional(GRU(units=lstm_layers[layer], return_sequences=True, activation=activation,
#                                         recurrent_activation=recurrent_activation,
#                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                                         dropout=dropout_rate[layer], recurrent_dropout=0.0)))
#
#         # receives LSTM with return sequences =True
#
#         # add attention
#         # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
#         model.add(attention())
#         # a, context = attention()(model)
#         # add denses
#         for layer in range(len(dense_layers)):
#             model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate_dense[layer]))
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(number_classes, activation='softmax'))
#         model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#         print(model.summary())
#
#         return model
#
#
# def bilstm_attention_embedding(input_dim, number_classes, output_dim,
#                                optimizer='Adam',
#                                lstm_layers=(64, 64, 32),
#                                activation='tanh',
#                                recurrent_activation='sigmoid',
#                                dropout_rate=(0.1, 0.1, 0.1),
#                                l1=1e-5, l2=1e-4,
#                                dense_layers=(32, 16),
#                                dropout_rate_dense=(0.1, 0.1),
#                                dense_activation="relu", loss='sparse_categorical_crossentropy'):
#     with strategy.scope():
#         model = Sequential()
#         # input dim timesteps = seq size , features. 21 features per character
#         model.add(Input(shape=(input_dim,), dtype='float32', name='main_input'))
#         # add initial dropout
#         model.add(Masking(mask_value=0))
#         model.add(Embedding(input_dim=len(alphabet) + 1, output_dim=output_dim))
#         for layer in range(len(lstm_layers)):
#             model.add(Bidirectional(
#                 LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
#                      recurrent_activation=recurrent_activation,
#                      kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                      dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
#             # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
#             #          recurrent_activation=recurrent_activation,
#             #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#             #          dropout=dropout_rate[layer], recurrent_dropout=0.0)
#
#         # receives LSTM with return sequences =True
#
#         # add attention
#         # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
#         model.add(attention())
#         # a, context = attention()(model)
#         # add denses
#         for layer in range(len(dense_layers)):
#             model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate_dense[layer]))
#
#         # Add Classification Dense, Compile model and make it ready for optimization
#         model.add(Dense(number_classes, activation='softmax'))
#         model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#         print(model.summary())
#         return model


# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(fps_x_hot, fps_y_encoded, test_size=0.2, val_size=0.2)
#
# vector_size = x_train.shape[1]
# final_units = fps_y_hot.shape[1]
# # model = KerasClassifier(build_fn=bilstm_simple, input_dim=vector_size, number_classes=final_units)
# # model = KerasClassifier(build_fn=lstm_simple, input_dim=vector_size, number_classes=final_units)
# # model = KerasClassifier(build_fn=gru_simple, input_dim=vector_size, number_classes=final_units)
# model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units,
#                         n_features=21)
# # model = KerasClassifier(build_fn=lstm_attention, input_dim=vector_size, number_classes=final_units)
# # model = KerasClassifier(build_fn=gru_attention, input_dim=vector_size, number_classes=final_units)
# # model = KerasClassifier(build_fn=bilstm_attention_embedding, input_dim=vector_size, number_classes=final_units,
# #                         output_dim=5)
#
# fps_x_hot = fps_x_hot.astype(np.int8)
#
# dl = DeepML(x_train=x_train, y_train=y_train, x_test=x_test.astype(np.int8),
#             y_test=y_test,
#             number_classes=final_units, problem_type='multiclass',
#             x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
#             path=dl_path,
#             report_name=report_name, validation_split=0.2,
#             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#             early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#             verbose=1)
#
# print(fps_x_hot.shape)
# test = fps_x_hot.shape[0] / 5
# train = fps_x_hot.shape[0] - test
# val = train * 0.3
# train = train - val
# print(train)
# print(test)
# print(val)
#
#
# model = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()
# print(scores)
# print(report)
# print(cm)
# dl.save_model(model_name)
# model = dl.get_model()
# model.model.save(model_name)
#
# model = load_model(model_name)
# model.summary()

# scores = dl.train_model_cv(x_cv=fps_x_hot.astype(np.float32), y_cv=fps_y_encoded.astype(np.float32), cv=5, model=model)

# print(fps_x)
# dl = DeepML(x_train=fps_x, y_train=fps_y_encoded, x_test=x_test.astype(np.int8),
#             y_test=y_test,
#             number_classes=final_units, problem_type='multiclass',
#             x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
#             path=dl_path,
#             report_name=report_name, validation_split=0.2,
#             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#             early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#             verbose=1)
# scores = dl.train_model_cv(x_cv=fps_x.astype(np.float32), y_cv=fps_y_encoded.astype(np.float32), cv=5, model=model)
#
# print(scores)
#
# K.clear_session()
# tf.keras.backend.clear_session()



# def deep_learning(x_train, x_test, y_train, y_test, x_dval, y_dval, path, report_name):
#
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     import logging
#     logging.disable(logging.WARNING)
#     logging.getLogger("tensorflow").setLevel(logging.ERROR)
#     os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.debugging.set_log_device_placement(True)
#     if gpus:
#         try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#                 # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')
#
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(e)
#
#     # # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
#     # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#     # OPTIMIZER
#     # # You can use a learning rate schedule to modulate how the learning rate of your optimizer changes over time:
#
#     lr_schedule = optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=1e-2,
#         decay_steps=10000,
#         decay_rate=0.9)
#     # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
#     # optimizer = keras.optimizers.SGD(learning_rate=0.001)
#     # optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
#     # optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
#     # optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
#     optimizer = optimizers.Adam(learning_rate=0.001)
#
#     ####################################################################3
#
#     # open deeplearning class
#     dl = DeepML(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
#                 number_classes=2, problem_type='binary',
#                 x_dval=x_dval, y_dval=y_dval,
#                 epochs=100, batch_size=512,
#                 reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#                 early_stopping_patience=30, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#                 path=path,
#                 report_name=report_name)
#
#     input_dim = x_train.shape[1]
#
#     # run one model
#     dnn_simple = dl.run_dnn_simple(input_dim, cv=5, optType='gridSearch', param_grid=None, n_iter_search=15, n_jobs=1,
#                                    scoring=make_scorer(matthews_corrcoef))
#     tensorflow.Keras.clear_session()
#     tensorflow.keras.backend.clear_session()
#
#     report_ml = str(report_name + '.txt')
#     path_for_model = str(report_name + '.sav')
#     path_precision_recall = str(report_name + '_precision_recall.png')
#     path_for_roc = str(report_name + '_roc.png')
#     path_for_val_curve = str(report_name + '_valcurve.png')
#     path_for_learn_curve = str(report_name)
#
#     dl.save_model(path=path_for_model)
#     reconstructed_model = dl.load_model(path=path_for_model)
#     score = dl.model_simple_evaluate(x_test=None, y_test=None, model=None)  # use the x_test and model saved in the class
#     scores, report, cm, cm2 = dl.model_complete_evaluate()
#     dl.precision_recall_curve(show=True, path_save=path_precision_recall)
#     dl.roc_curve(path_save=path_for_roc, show=True)
#     dl.plot_validation_curve(param_name='dropout_rate',
#                              param_range=[0.1, 0.3, 0.2, 0.4],
#                              cv=5, score=make_scorer(matthews_corrcoef), title="Validation Curve dnn simple",
#                              show=True,
#                              path_save=path_for_val_curve)  # todo fix? do not know if it makes sense
#     dl.plot_learning_curve(title='Learning curve', cv=5,
#                            path_save=path_for_learn_curve, show=True, scalability=True, performance=True)
#
#
#
