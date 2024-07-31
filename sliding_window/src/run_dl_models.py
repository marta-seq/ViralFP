import os
import pandas as pd
import numpy as np
from tensorflow import keras
import random
import tensorflow as tf

from tensorflow.keras import optimizers
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers, Input
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

from propythia_src.deep_ml import DeepML

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# print(tf.executing_eagerly())
# print('execution eager')



def run_dl(x_train, x_test, y_train, y_test, x_dval, y_dval, model, param_grid, path, report_name,
           optType='randomizedSearch', cv=10, n_iter=15):

    ####################################################################

    # names for files

    report_name = str(report_name + '.txt')
    path_for_model = str(report_name + '.sav')
    path_precision_recall = str(report_name + '_precision_recall.png')
    path_for_roc = str(report_name + '_roc.png')
    path_for_val_curve = str(report_name + '_valcurve.png')
    path_for_learn_curve = str(report_name)

    ####################################################################

    # open deep learning class
    dl = DeepML(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                number_classes=2, problem_type='binary',
                x_dval=x_dval, y_dval=y_dval,
                epochs=150, batch_size=64,
                validation_split=0.2,
                reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                path=path,
                report_name=report_name,
                verbose=1)


    # # # run one model predefined with gridsearch default

    scores_cv = dl.train_model_cv(x_cv=x_train, y_cv=y_train, cv=cv, model=model)
    model = dl.run_model(model=model)
    #
    #
    # model_optimized = dl.get_opt_params(param_grid, model, optType=optType, cv=cv, dataX=x_train,
    #                                     datay=y_train,
    #                                     n_iter_search=n_iter, n_jobs=1,
    #                                     scoring=make_scorer(matthews_corrcoef))
    # dnn = dl.run_dnn_simple(
    #                    input_dim = 483,
    #                    optimizer='Adam',
    #                    hidden_layers=(128, 64),
    #                    dropout_rate=(0.3,),
    #                    batchnormalization=(True,),
    #                    l1=1e-5, l2=1e-4,
    #                    final_dropout_value=0.3,
    #                    initial_dropout_value=0.0,
    #                    loss_fun=None, activation_fun=None,
    #                    cv=3, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
    #                    scoring=make_scorer(matthews_corrcoef))


    score = dl.model_simple_evaluate(x_test=x_test, y_test=y_test)
    scores, report, cm, cm2 = dl.model_complete_evaluate(x_test=x_test, y_test=y_test)
    print(scores_cv)
    print(score)
    print(scores)
    print(report)
    print(cm)

    # # save model
    # dl.save_model(path=path_for_model)
    #
    # # model = dl.get_model()
    # # model.model.save(path_for_model)
    # # model = dl.load_model(path_for_model)
    # model_optimized.summary()
    #
    # # curves
    # dl.precision_recall_curve(show=True, path_save=path_precision_recall)
    # dl.roc_curve(path_save=path_for_roc, show=True)
    # dl.plot_validation_curve(param_name='dropout_rate',
    #                          param_range=[0.1, 0.3, 0.2, 0.4],
    #                          cv=5, score=make_scorer(matthews_corrcoef), title="Validation Curve dnn simple",
    #                          show=True,
    #                          path_save=path_for_val_curve)  # todo fix? do not know if it makes sense
    # dl.plot_learning_curve(title='Learning curve', cv=5,
    #                        path_save=path_for_learn_curve, show=True, scalability=True, performance=True)
    #
    # # clear session
    # tf.keras.backend.clear_session()
    #



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

# print(scores)
#
# tf.keras.clear_session()
# tf.keras.backend.clear_session()
