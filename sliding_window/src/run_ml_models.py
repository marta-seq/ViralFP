import pickle
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
import tensorflow
from tensorflow.keras import optimizers
from propythia_src.deep_ml import DeepML
from propythia_src.feature_selection import FeatureSelection
from propythia_src.shallow_ml import ShallowML

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


def run_ml(df_x_train, df_x_test, y_train, y_test, model, report, **params):
    # standard scaler based on train
    scaler = StandardScaler()
    scaler.fit(df_x_train)
    df_x_train_std = scaler.transform(df_x_train)
    df_x_test_std = scaler.transform(df_x_test)

    report_ml = str(report + '.txt')
    path_for_roc = str(report + '_roc.png')
    path_for_val_curve = str(report + '_valcurve.png')
    path_for_learn_curve = str(report)
    path_fi = str(report + '_fi.png')

    ml = ShallowML(x_train=df_x_train_std, x_test=df_x_test_std, y_train=y_train, y_test=y_test,
                   report_name=report_ml, columns_names=df_x_train.columns)

    try:
        best_lsvm = ml.train_best_model(model_name=model, model=None, score=make_scorer(matthews_corrcoef), cv=10,
                                        optType='gridSearch', param_grid=None,
                                        n_jobs=10, random_state=1, n_iter=51, **params)
        filename = str(report + '.sav')

        pickle.dump(best_lsvm, open(filename, 'wb'))

        scores, report, cm, cm2 = ml.score_testset()
        print(scores)
        print(report)
        print(cm)

        try:
            ml.plot_roc_curve(title='ROC curve for ' + str(model),
                              path_save=path_for_roc, show=False)
        except Exception as e:
            print(str(e))

        try:
            ml.plot_learning_curve(n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5),
                                   path_save=path_for_learn_curve, show=False, scalability=True,
                                   performance=True)
        except Exception as e:
            print(str(e))
        try:
            ml.features_importances_df(top_features=30, model_name=model)
        except Exception as e:
            print(str(e))
        try:
            ml.features_importances_plot(top_features=20, show=False, model_name=model,
                                         path_save=path_fi, column_to_plot=0,
                                         kind='barh', figsize=(9, 7), color='r', edgecolor='black')
        except Exception as e:
            print(str(e))

    except Exception as e:
        print('error running' + str(model))
        print(str(e))

    #     try:
    #         param_name = 'clf__C'
    #     param_range = [0.001, 0.01, 0.1, 1.0, 10, 100]
    #     ml.plot_validation_curve(param_name=param_name, param_range=param_range, n_jobs=10, show=False,
    #                              path_save=path_for_val_curve)
    # except Exception as e:
    #     print(str(e))



def run_ml2(df_x_total, y_total, df_x_test, y_test, model, report, **params):

    report_ml = str(report + '.txt')
    path_for_roc = str(report + '_roc.png')
    path_for_val_curve = str(report + '_valcurve.png')
    path_for_learn_curve = str(report)
    path_fi = str(report + '_fi.png')

    ml = ShallowML(x_train=df_x_total, x_test=df_x_test, y_train=y_total, y_test=y_test,
                   report_name=report_ml, columns_names=df_x_total.columns)

    try:
        best_lsvm = ml.train_best_model(model_name=model, model=None, score=make_scorer(matthews_corrcoef), cv=10,
                                        optType='gridSearch', param_grid=None,
                                        n_jobs=10, random_state=1, n_iter=51, **params)
        filename = str(report + '.sav')

        pickle.dump(best_lsvm, open(filename, 'wb'))

        scores, report, cm, cm2 = ml.score_testset()
        print(scores)
        print(report)
        print(cm)

        try:
            ml.plot_roc_curve(title='ROC curve for ' + str(model),
                              path_save=path_for_roc, show=False)
        except Exception as e:
            print(str(e))

        try:
            ml.plot_learning_curve(n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5),
                                   path_save=path_for_learn_curve, show=False, scalability=True,
                                   performance=True)
        except Exception as e:
            print(str(e))
        try:
            ml.features_importances_df(top_features=30, model_name=model)
        except Exception as e:
            print(str(e))
        try:
            ml.features_importances_plot(top_features=20, show=False, model_name=model,
                                         path_save=path_fi, column_to_plot=0,
                                         kind='barh', figsize=(9, 7), color='r', edgecolor='black')
        except Exception as e:
            print(str(e))

    except Exception as e:
        print('error running' + str(model))
        print(str(e))
