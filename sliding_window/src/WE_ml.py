import pandas as pd
import numpy as np
import os
import sys
import time
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from ppropythia.shallow_ml import ShallowML
from propythia.sequence import sub_seq_sliding_window
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

seed = 42
np.random.seed(seed)
random.seed(seed)

#get the dataset of the segments

all = 'datasets/all_cluster.csv'
third = 'datasets/third_cluster.csv'
half = 'datasets/half_cluster.csv'

dataframe_data = pd.read_csv(all)
path = 'results/ml'
report_name = 'all_Window21_gap1_WEPROTVEC_10groupedKFOLD_cluster80_rf_m3'
model_name = 'rf'
class_weight = None # 'balanced' # None #
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)

# transform then using protvec
# approach 1 sum all vectors and do ml
from propythia.Bumblebee.src.word_embedding_main import WordEmbedding as wv
t = time.time()
# this method 3 I need every seq to be on the same len
max_len = 21
seq_pad = []
for seq in dataframe_data['seq']:
    l = len(seq)
    if l<max_len:
        new_seq = seq
        # new_seq = seq + 'X'*(max_len - l)
    else:
        new_seq = seq[:max_len]
    seq_pad.append(new_seq)
w2v = wv(emb_matrix_file='propythia/Bumblebee/src/protVec_100d_3grams.csv',
         ngram_len=3, sequence_max_len=21, vectordim=100)
#seqs_vector = w2v.convert_sequences2vec(method=1,sequences=seqs_new,padding=True, array_flat=True)
# seqs_vector = []
# for i in dataframe_data['seq']:
#     vector = w2v.convert_seq2vec(method=1, sequence=i, padding = True)
#     seqs_vector.append(vector)

seqs_vector = w2v.convert_sequences2vec(method=3, sequences=seq_pad, array_flat=True, padding=True)
print('seqs vectorized')
X = np.array(seqs_vector)
print(X.shape)
t2 = time.time()
print('seconds to WE', t2 - t)


# def kfold split but agglomerating sequences similar
groups = dataframe_data['cluster80']  # cluster90
y = dataframe_data['label']  # to stratidy based on that

# MACHINE LEARNING
# https://towardsdatascience.com/cross-validate-on-data-containing-groups-correctly-ffa7173a37e6
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
# sgkf.get_n_splits(X=X, y=y, groups=groups)

ml = ShallowML(x_train=X, x_test=None, y_train=y, y_test=None, report_name=REPORT,
               columns_names=None,
               problem_type='src', classifier = None)

# train best model choosing best parameters
ml.train_best_model(model_name=model_name, model=None, scaler=None,
                    score=make_scorer(matthews_corrcoef),
                    cv=sgkf, optType='gridSearch', param_grid=None,
                    n_jobs=10,
                    random_state=1, n_iter=15, refit=True,  # refit on whole dataset with the best model
                    groups = groups # groups for the grid search
                    # ,class_weight = class_weight, # cannot entry in gboosting
                    # probability = True # for svc
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
test_csv = pd.read_csv('ViralFP_dataset/data/holdout_dataset.csv')

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
    # seqs_vector_test = []
    # for i in df['seq']:
    #     vector = w2v.convert_seq2vec(method=1, sequence=i)
    #     seqs_vector_test.append(vector)
    # X_test = np.array(seqs_vector_test)
    X_test = w2v.convert_sequences2vec(method=3, sequences=df['seq'], array_flat=True)

    # predict
    predict_df = ml.predict(X_test)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))
