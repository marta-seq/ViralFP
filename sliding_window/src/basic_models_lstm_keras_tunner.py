import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool3D,TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from propythia_new.src.propythia.attention import attention

import keras_tuner
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

# https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
# https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

def bilstm_builder(hp):

    timestep = 21
    dim= 21
    loss_fun='binary_crossentropy'
    activation_fun = 'sigmoid'
    final_units = 1
    l1=1e-5
    l2=1e-4
    # with strategy.scope():
    model = tf.keras.Sequential()
    # (batch_size, timesteps, dim)
    model.add(Input(shape=(timestep, dim), dtype='float32', name='main_input'))

    # tune number of lstm
    n_layers_lstm = hp.Int("num_layers", 1, 2)

    for i in range(n_layers_lstm - 1):
        name = str('units_' + str(i))
        name_dr = str('dropout_' + str(i))
        hp_units_mid = hp.Int(name, min_value=8, max_value=64, step=8)
        dr_mid = hp.Float(name_dr, min_value=0.0, max_value=0.4, step=0.1)
        # add other lstm layer
        model.add(Bidirectional(LSTM(units=hp_units_mid, return_sequences=True,
                                         activation='tanh', recurrent_activation='sigmoid',
                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                         dropout=dr_mid, recurrent_dropout=dr_mid)))
    # add last lstm layer
    hp_units_last = hp.Int('units', min_value=8, max_value=64, step=8)
    dr_last = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
    model.add(Bidirectional(LSTM(units=hp_units_last, return_sequences=False,
                                     activation='tanh', recurrent_activation='sigmoid',
                                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                     dropout=dr_last, recurrent_dropout=dr_last)))

    # add denses
    n_layers_dense = hp.Int("num_layers_dense", 1, 2,1)

    for i in range(n_layers_dense):
        name = 'units_dense' + str(i)
        name_dr = 'dropout_dense' + str(i)
        hp_units_dense = hp.Int(name, min_value=8, max_value=64, step=16)
        dr_dense = hp.Float(name_dr, min_value=0.1, max_value=0.4, step=0.1)
        # dr_dense = 0.2
        model.add(Dense(units=hp_units_dense, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        model.add(BatchNormalization())
        model.add(Dropout(dr_dense))

    # Add Classification Dense, Compile model and make it ready for optimization
    model.add(Dense(final_units, activation=activation_fun))


    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(optimizer='adam', #keras.optimizers.Adam(learning_rate=hp_learning_rate
                  loss=loss_fun,
                  metrics=['accuracy', tf.keras.metrics.AUC()])# , tf.keras.metrics.AUC(from_logits=True)
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    # Without the line above AttributeError: 'dict' object has no attribute 'name'
    # https://github.com/tensorflow/tensorflow/issues/38988
    return model


def bilstm_attention_builder(hp):

    timestep = 21
    dim= 21
    loss_fun = 'binary_crossentropy'
    activation_fun = 'sigmoid'
    final_units = 1
    l1=1e-5
    l2=1e-4
    with strategy.scope():
        model = tf.keras.Sequential()
        # (batch_size, timesteps, dim)
        model.add(Input(shape=(timestep, dim), dtype='float32', name='main_input'))
        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(tf.keras.layers.Masking(mask_value=0))

        # tune number of lstm
        n_layers_lstm = hp.Int("num_layers", 1, 2)

        for i in range(n_layers_lstm):
            name = 'units_lstm' + str(i)
            name_dr = 'dr_lstm' + str(i)
            hp_units_mid = hp.Int(name, min_value=8, max_value=64, step=8)
            dr_mid = hp.Float(name_dr, min_value=0.0, max_value=0.4, step=0.1)
            # add other lstm layer
            model.add(Bidirectional(LSTM(units=hp_units_mid, return_sequences=True,
                                         activation='tanh', recurrent_activation='sigmoid',
                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                         dropout=dr_mid, recurrent_dropout=dr_mid)))


        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)

        # add denses
        n_layers_dense = hp.Int("num_layers_dense", 1, 2,1)

        for i in range(n_layers_dense):
            name = 'units_dense' + str(i)
            name_dr = 'dropout_dense' + str(i)
            hp_units_dense = hp.Int(name, min_value=8, max_value=256, step=32)
            dr_dense = hp.Float(name_dr, min_value=0.1, max_value=0.4, step=0.1)
            model.add(Dense(units=hp_units_dense, activation='relu',
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dr_dense))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(final_units, activation=activation_fun))


        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=loss_fun,
                      metrics=['accuracy'])
        model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

        # Without the line above AttributeError: 'dict' object has no attribute 'name'
        # https://github.com/tensorflow/tensorflow/issues/38988
        return model






def bilstm_embedding_builder(hp):

    timestep = 21
    dim= 21
    loss_fun='binary_crossentropy'
    activation_fun = 'sigmoid'
    final_units = 1
    l1=1e-5
    l2=1e-4
    with strategy.scope():
        model = tf.keras.Sequential()

        input_dim_emb=21
        input_length=21
        mask_zero=True
        # model.add(Input(shape=(input_dim,),dtype='float32', name='main_input'))
        # Add an Embedding layer
        hp_out_dim = hp.Choice('embedding_output', values=[5, 11, 21])

        model.add(Embedding(
            input_dim=input_dim_emb, output_dim=hp_out_dim,
            input_length=input_length,
            mask_zero=mask_zero))
        # model.add(Flatten(data_format=None))
        # model.add(tf.keras.layers.Reshape((input_length,output_dim)))

        # tune number of lstm
        n_layers_lstm = hp.Int("num_layers", 1, 4)

        for i in range(n_layers_lstm - 1):
            hp_units_mid = hp.Int('units', min_value=16, max_value=512, step=32)
            dr_mid = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
            # add other lstm layer
            model.add(Bidirectional(LSTM(units=hp_units_mid, return_sequences=True,
                                         activation='tanh', recurrent_activation='sigmoid',
                                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                         dropout=dr_mid, recurrent_dropout=dr_mid)))
        # add last lstm layer
        hp_units_last = hp.Int('units', min_value=8, max_value=64, step=8)
        dr_last = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
        model.add(Bidirectional(LSTM(units=hp_units_last, return_sequences=False,
                                     activation='tanh', recurrent_activation='sigmoid',
                                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                     dropout=dr_last, recurrent_dropout=dr_last)))

        # add denses
        n_layers_dense = hp.Int("num_layers_dense", 1, 3,1)

        for layer in range(n_layers_dense):
            hp_units_dense = hp.Int('units', min_value=8, max_value=256, step=32)
            dr_dense = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
            dr_dense = 0.2
            model.add(Dense(units=hp_units_dense, activation='relu',
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dr_dense))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(final_units, activation=activation_fun))


        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=loss_fun,
                      metrics=['accuracy'])
        model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

        # Without the line above AttributeError: 'dict' object has no attribute 'name'
        # https://github.com/tensorflow/tensorflow/issues/38988
        return model



# def create_lstm_embedding(number_classes,
#                           optimizer='Adam',
#                           input_dim_emb=21, output_dim=128, input_length=1000, mask_zero=True,
#                           bilstm=True,
#                           lstm_layers=(128, 64),
#                           activation='tanh',
#                           recurrent_activation='sigmoid',
#                           dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
#                           l1=1e-5, l2=1e-4,
#                           dense_layers=(64, 32),
#                           dense_activation="relu",
#                           dropout_rate_dense = (0.3,),
#                           batchnormalization = (True,),
#                           loss_fun='binary_crossentropy', activation_fun='sigmoid'):
#
#     if len([dropout_rate]) == 1:
#         dropout_rate = list(dropout_rate * len(lstm_layers))
#     print(dropout_rate)
#
#     if len([recurrent_dropout_rate]) == 1:
#         recurrent_dropout_rate = list(recurrent_dropout_rate * len(lstm_layers))
#     if len([dropout_rate_dense]) == 1:
#         dropout_rate_dense = list(dropout_rate_dense * len(dense_layers))
#     print(dropout_rate_dense)
#     if len([batchnormalization]) == 1:
#         batchnormalization = list(batchnormalization * len(dense_layers))
#
#
#     last_lstm_layer = lstm_layers[-1]
#
#     with strategy.scope():
#         model = Sequential()
#         # model.add(Input(shape=(input_dim,),dtype='float32', name='main_input'))
#         # Add an Embedding layer expecting input vocab of size 1000, and
#         model.add(
#             Embedding(input_dim=input_dim_emb, output_dim=output_dim, input_length=input_length, mask_zero=mask_zero))
#         # model.add(Flatten(data_format=None))
#         # model.add(tf.keras.layers.Reshape((input_length,output_dim)))
#         # print(model.output_shape) #None 100, 256
#         last_layer = LSTM(units=last_lstm_layer, return_sequences=False,
#                           activation=activation, recurrent_activation=recurrent_activation,
#                           dropout=dropout_rate[-1],
#                           recurrent_dropout=recurrent_dropout_rate[-1],
#                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))
#         if not bilstm:
#             # add lstm layers
#             for layer in range(len(lstm_layers) - 1):
#                 model.add(LSTM(units=lstm_layers[layer], return_sequences=True,
#                                activation=activation, recurrent_activation=recurrent_activation,
#                                dropout=dropout_rate[layer],
#                                recurrent_dropout=recurrent_dropout_rate[layer],
#                                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             # add last lstm layer
#             model.add(last_layer)
#
#         elif bilstm:
#             # add other lstm layer
#             for layer in range(len(lstm_layers) - 1):
#                 model.add(Bidirectional(LSTM(units=lstm_layers[layer], return_sequences=True,
#                                              activation=activation, recurrent_activation=recurrent_activation,
#                                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
#                                              dropout=dropout_rate[layer], recurrent_dropout=recurrent_dropout_rate[layer])))
#             # add last lstm layer
#             model.add(Bidirectional(last_layer))
#

#
#
#         # add denses
#         for layer in range(len(dense_layers)):
#             model.add(Dense(units=dense_layers[layer], activation=dense_activation,
#                             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
#             if batchnormalization[layer]:
#                 model.add(BatchNormalization())
#             model.add(Dropout(dropout_rate_dense[layer]))
#         # Add Classification Dense, Compile model and make it ready for optimization
#         # model binary
#         model.add(Dense(number_classes, activation=activation_fun))
#         model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy', 'val_loss'])
#         return model



# todo add GRU? LSTM













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
                     dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, n_features,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
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
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model



    # Parametros final primeiras tentativas
    # timestep = 21
    # dim= 21
    # loss_fun='binary_crossentropy'
    # activation_fun = 'sigmoid'
    # final_units = 1
    # l1=1e-5
    # l2=1e-4
    # # with strategy.scope():
    # model = tf.keras.Sequential()
    # # (batch_size, timesteps, dim)
    # model.add(Input(shape=(timestep, dim), dtype='float32', name='main_input'))
    # # # # add initial dropout
    # # if hp.Boolean("dropout"):
    # #     model.add(Dropout(0.1))
    # # dr_initial = hp.Float(name='dropout', min_value=0.0, max_value=0.1, step=0.1)
    # # model.add(Dropout(dr_initial))
    #
    # # tune number of lstm
    # n_layers_lstm = hp.Int("num_layers", 1, 4)
    #
    # for i in range(n_layers_lstm - 1):
    #     hp_units_mid = hp.Int('units', min_value=16, max_value=512, step=32)
    #     dr_mid = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
    #     # add other lstm layer
    #     model.add(Bidirectional(LSTM(units=hp_units_mid, return_sequences=True,
    #                                  activation='tanh', recurrent_activation='sigmoid',
    #                                  kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #                                  dropout=dr_mid, recurrent_dropout=dr_mid)))
    # # add last lstm layer
    # hp_units_last = hp.Int('units', min_value=8, max_value=64, step=8)
    # dr_last = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
    # model.add(Bidirectional(LSTM(units=hp_units_last, return_sequences=False,
    #                              activation='tanh', recurrent_activation='sigmoid',
    #                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #                              dropout=dr_last, recurrent_dropout=dr_last)))
    #
    # # add denses
    # n_layers_dense = hp.Int("num_layers_dense", 1, 3,1)
    #
    # for layer in range(n_layers_dense):
    #     hp_units_dense = hp.Int('units', min_value=8, max_value=256, step=32)
    #     dr_dense = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
    #     # dr_dense = 0.2
    #     model.add(Dense(units=hp_units_dense, activation='relu',
    #                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(dr_dense))
    #
    # # Add Classification Dense, Compile model and make it ready for optimization
    # model.add(Dense(final_units, activation=activation_fun))
    #
    #
    # # Tune the learning rate for the optimizer
    # # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    # # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    #
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
    #               loss=loss_fun,
    #               metrics=['accuracy'])
    # model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]
    #
    # # Without the line above AttributeError: 'dict' object has no attribute 'name'
    # # https://github.com/tensorflow/tensorflow/issues/38988
    # return model


    ####################### no optimization

def bilstm():

    timestep = 23 #23  #19 Protvec
    dim= 1280#320   # 100 Protvec
    loss_fun='binary_crossentropy'
    activation_fun = 'sigmoid'
    final_units = 1
    l1=1e-5
    l2=1e-4
    # with strategy.scope():
    model = tf.keras.Sequential()
    # (batch_size, timesteps, dim)
    model.add(Input(shape=(timestep, dim), dtype='float32', name='main_input'))


    model.add(Bidirectional(LSTM(units=128, return_sequences=True,
                                 activation='tanh', recurrent_activation='sigmoid',
                                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                                 dropout=0.2, recurrent_dropout=0.2)))

    model.add(Bidirectional(LSTM(units=64, return_sequences=True,
                   activation='tanh', recurrent_activation='sigmoid',
                   kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                   dropout=0.2, recurrent_dropout=0.2)))

    # model.add(Bidirectional(LSTM(units=8, return_sequences=False,
    #                              activation='tanh', recurrent_activation='sigmoid',
    #                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #                              dropout=0.2, recurrent_dropout=0.2)))

    model.add(attention())

    # add dense
    model.add(Dense(units=64, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Add Classification Dense, Compile model and make it ready for optimization
    model.add(Dense(final_units, activation=activation_fun))

    model.compile(optimizer='adam', #keras.optimizers.Adam(learning_rate=hp_learning_rate
                  loss=loss_fun,
                  metrics=['accuracy'])# , tf.keras.metrics.AUC(from_logits=True)
    model._layers = [layer for layer in model._layers if not isinstance(layer, dict)]

    # Without the line above AttributeError: 'dict' object has no attribute 'name'
    # https://github.com/tensorflow/tensorflow/issues/38988
    return model