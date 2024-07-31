import pandas as pd
import numpy as np
import os
import sys
import time
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
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
path = 'results/dl'
report_name = 'all_Window21_gap1_WEPROTVEC_method1_10groupedKFOLD_cluster80_lstm6'

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

seqs_vector = w2v.convert_sequences2vec(method=1, sequences=seq_pad, array_flat=False, padding=True)
print('seqs vectorized')
X = np.array(seqs_vector)
print(X.shape)
t2 = time.time()
print('seconds to WE', t2 - t)


# def kfold split but agglomerating sequences similar
groups = dataframe_data['cluster80']  # cluster90
y = dataframe_data['label']  # to stratidy based on that
# DL using o keras Tuner
from basic_models_lstm_keras_tunner import bilstm_builder, bilstm_attention_builder
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', classes= np.unique(y),y = y)
class_weights = {i:w for i,w in enumerate(class_weights)}
input_dim = 19
print(X.shape)

from propythia.deep_ml import DeepML

dl = DeepML(x_train=X, y_train=y,
            x_test=None, y_test=None,
            number_classes=2, problem_type='binary',
            x_dval=None, y_dval=None,
            model=None,
            epochs=500, batch_size=512,
            callbacks=None,
            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
            early_stopping_patience=30, reduce_lr_patience=30, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
            path=path,
            report_name=report_name,
            verbose=1, validation_split=0.1,
            shuffle=True, class_weights=class_weights)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

from sklearn.model_selection import cross_validate
from sklearn import metrics

from basic_models_lstm_keras_tunner import bilstm

scoring = {'accuracy': make_scorer(metrics.accuracy_score),
           'prec': 'precision',
           'recall': 'recall',
           'MCC': make_scorer(matthews_corrcoef),
           'F1': make_scorer(metrics.f1_score),
           'roc_auc': make_scorer(metrics.roc_auc_score)
           }
print('start cross validate ')

scores = dl.cross_validate(bilstm, cv=sgkf,scoring=scoring, return_train_score=False, groups=groups)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# scores = dl.train_model_cv(x_cv=X, y_cv=y, cv=sgkf, model=None, groups=groups, refit=True)
print(scores)

# model
# run the model
model = dl.run_model(model=bilstm()) # KerasClassifier(build_fn=bilstm)


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
    X_test = w2v.convert_sequences2vec(method=1, sequences=df['seq'], array_flat=False, padding=True)

    # predict
    predict_df = dl.predict(X_test)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))
