import pandas as pd
import numpy as np
import os
import sys
import time
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# sys.path.append('viral_fp/')

from propythia.sequence import ReadSequence
from propythia.protein_descriptors import ProteinDescriptors
from propythia.preprocess import Preprocess
from propythia.shallow_ml import ShallowML
from propythia.sequence import sub_seq_sliding_window
from propythia.feature_selection import FeatureSelection

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

seed = 42
np.random.seed(seed)
random.seed(seed)

all = 'datasets/all_cluster.csv'
third = 'datasets/third_cluster.csv'
half = 'datasets/half_cluster.csv'

dataframe_data = pd.read_csv(third)
path = '/sliding_window/results'
report_name = 'third_Window21_gap1_PHYSALL_NONE_10groupedKFOLD_cluster80_linear_svc_try11'
model_name = 'linear_svc'
feat_sel = 'mutual'
class_weight =  None #'balanced' # None #
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)


# preprocessing sequences if they have strange characters
read_seqs = ReadSequence()
dataframe_data = read_seqs.par_preprocessing(dataset=dataframe_data, col='seq', B='N', Z='Q', U='C', O='K', J='I', X='')

# calculate descriptors using multiprocessing
descriptors_df = ProteinDescriptors(dataset=dataframe_data, col='seq')
# calculate all

t = time.time()
df = descriptors_df.get_all(maxlag_qso=10,maxlag_socn=10, tricomp=True)
# df = descriptors_df.get_adaptable([17, 21, 25, 29, 30, 31, 36], maxlag_qso=10, maxlag_socn=10)
t2 = time.time()
print('seconds to descriptors', t2 - t)
print(df.shape)

df2 = df.drop(['seq', 'label', 'type','cluster90', 'cluster80', 'cluster70'], axis=1)
p = Preprocess()
dataset_clean, columns_deleted = p.preprocess(df2, columns_names=True, threshold=0, standard=False)

# columns_deleted length=15510

df_prep = df.drop(columns_deleted, axis=1)

print(df_prep.shape)
#
# def feat_select(df_x_train, df_y_train, type, report_name_fselect):
#
#     from sklearn.svm import LinearSVC
#     from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
#     from sklearn.linear_model import LogisticRegression
#
#
#
#
#     # 3. methods of feature selection
#
#     fselect = FeatureSelection(x_original=df_x_train_std, target=df_y_train,
#                                columns_names=df_x_train.columns, report_name=report_name_fselect)
#     column_selected = []
#     if type == 'mutual':
#         # mutual  Percentile ---> Percent of features to keep.
#         transformer, x_fit_univariate, x_transf_model, column_selected, scores, scores_df = \
#             fselect.run_univariate(score_func=mutual_info_classif, mode='percentile',
#                                    param=50)
#     else:
#         if type == 'svc':
#             m = LinearSVC(C=1, penalty="l1", dual=False)
#         elif type == 'tree':
#             m = ExtraTreesClassifier(n_estimators=150)
#             m = RandomForestClassifier(n_estimators=150)
#         else:
#             m = LogisticRegression()
#         select_model, x_transf_model, column_selected, feat_impo, feat_impo_df = \
#             fselect.run_from_model(model=m)
#
#     print('number of columns', column_selected.shape)
#
#     # 5. Check everything
#     print('x_train_old', df_x_train.shape, '\n',
#           'number_columns_selected', len(column_selected))
#
#     return column_selected

# def kfold split but agglomerating sequences similar
groups = df_prep['cluster80']  # cluster90
X = df_prep.drop(['seq', 'label', 'type', 'cluster90', 'cluster80', 'cluster70'], axis=1)
y = df_prep['label']  # to stratidy based on that

# scaling
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

if feat_sel:
    from sklearn.feature_selection import mutual_info_classif
    report_name_fselect = report_name + 'fsel_mut'
    # 3. methods of feature selection

    fselect = FeatureSelection(x_original=X_std, target=y,
                               columns_names=X.columns, report_name=report_name_fselect)
    column_selected = []
    # if type == 'mutual':
    # mutual  Percentile ---> Percent of features to keep.
    transformer, x_fit_univariate, x_transf_model, column_fsel, scores, scores_df = \
        fselect.run_univariate(score_func=mutual_info_classif, mode='percentile',
                               param=50)

    X_std = x_transf_model
    print('fsel')
    print(X.columns[column_fsel])





# MACHINE LEARNING
# https://towardsdatascience.com/cross-validate-on-data-containing-groups-correctly-ffa7173a37e6
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
# sgkf.get_n_splits(X=X, y=y, groups=groups)

ml = ShallowML(x_train=X_std, x_test=None, y_train=y, y_test=None, report_name=REPORT,
               columns_names=X.columns,
                problem_type='src', classifier = None)

# train best model choosing best parameters
ml.train_best_model(model_name=model_name, model=None, scaler=None,
                 score=make_scorer(matthews_corrcoef),
                 cv=sgkf, optType='gridSearch', param_grid=None,
                 n_jobs=10,
                 random_state=1, n_iter=15, refit=True,  # refit on whole dataset with the best model
                groups = groups, # groups for the grid search
                # ,class_weight = class_weight, # cannot entry in gboosting
                # probability = True) # for svc)
                    )


from sklearn.model_selection import cross_validate
from sklearn import metrics
scoring = {'accuracy': make_scorer(metrics.accuracy_score),
           'prec': 'precision',
           'recall': 'recall',
           'MCC': make_scorer(matthews_corrcoef),
           'F1': make_scorer(metrics.f1_score),
           'roc_auc': make_scorer(metrics.roc_auc_score)
           }
# scikit learn
# clf = ml.get_model()
# # scores = cross_validate(clf, X_std, y, cv=sgkf,
#                         scoring=scoring, return_train_score=False, groups=groups)
# print(pd.DataFrame(scores))

# add to propythia just because reports
scores = ml.cross_validate(cv=sgkf,scoring=scoring, return_train_score=False, groups=groups)
print(scores)



############################### TESTING SEQUENCES
test_csv = pd.read_csv('../ViralFP_dataset/data/holdout_dataset.csv')

for i in range(len(test_csv)):
    seq_fuso = test_csv['Sequence_fusogenic'][i]
    idProtein = test_csv['idProtein'][i]
    name = test_csv['Name'][i]

    # create subsequences
    list_seq, indices = sub_seq_sliding_window(ProteinSequence=seq_fuso,
                                               window_size=21, gap=1)
    # calculates all features
    df = pd.DataFrame()
    df['seq']= list_seq
    df['indices'] = indices

    descriptors_df = ProteinDescriptors(dataset=df, col='seq')
    df = descriptors_df.get_all(maxlag_qso=10,maxlag_socn=10, tricomp=True)

    # select columns that are in X
    df_prep = df.drop(columns_deleted, axis=1)
    X_test = df[X.columns]
    # select the columns that are selcted in X
    # standardized c o X_train
    X_test_std = scaler.transform(X_test)

    if feat_sel:
        X_test = transformer.transform(X_test_std)
    # X_test = df_prep.drop(['seq', 'indices'], axis=1)

    # predict
    predict_df = ml.predict(X_test)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))

ml.features_importances_plot(classifier=None, top_features=20, model_name=None,
                          column_to_plot=None,
                          show=True, path_save='feat_impo.png',
                          title=None,
                          kind='barh', figsize=(9, 7), color='r', edgecolor='black')

feat = ml.features_importances_df(classifier=None, model_name=None, top_features=20, column_to_sort='mean_coef')

print(feat)