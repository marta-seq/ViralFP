import pandas as pd
import numpy as np
import os
import sys
import time
import random

import logging
import warnings

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('/home/martinha/PycharmProjects/viral_fp/')

from propythia_new.src.propythia.sequence import ReadSequence
from propythia_new.src.propythia.protein_encoding import Encoding
from propythia_new.src.propythia.deep_ml import DeepML

from propythia_new.src.propythia.sequence import sub_seq_sliding_window

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

import keras_tuner

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

all = 'datasets/all_cluster.csv'
third = 'datasets/third_cluster.csv'
half = 'datasets/half_cluster.csv'

dataframe_data = pd.read_csv(half)
path = 'results/dl'
report_name = 'half_Window21_gap1_ENCODING_OHE_DL_10groupedKFOLD_cluster80_LSTM_cweiV3'
model_name = ''

class_weight = None  # 'balanced' # None #

############################################################################
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)

# preprocessing sequences if they have strange characters
read_seqs = ReadSequence()
dataframe_data = read_seqs.par_preprocessing(dataset=dataframe_data, col='seq', B='N', Z='Q', U='C', O='K', J='I', X='')


def encode_sequence(sequences, seq_len, padding_truncating='post', encoding=''):
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # define a mapping of chars to integers
    alphabet = "XARNDCEQGHILKMFPSTWYV"
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    sequences_integer_ecoded = []
    for seq in sequences:
        # if 'X' not in alphabet:
        seq = seq.replace('X', '')  # unknown character eliminated
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)
    list_of_sequences_integer = pad_sequences(sequences_integer_ecoded, maxlen=seq_len, dtype='int32',
                                              padding=padding_truncating, truncating=padding_truncating, value=0.0)
    list_of_sequences_aa = []
    for seq in list_of_sequences_integer:
        pad_aa_list = [int_to_char[char] for char in seq]
        pad_aa = ''.join(pad_aa_list)
        list_of_sequences_aa.append(pad_aa)

    # if encoding == 'ohe':
    shape_hot = len(alphabet) * seq_len  # 20000
    fps_x_2d = to_categorical(list_of_sequences_integer)  # shape (samples, 1000,20)
    fps_x_2d = fps_x_2d.reshape(fps_x_2d.shape[0], fps_x_2d.shape[1], fps_x_2d.shape[2])
    fps_x_1d = pd.DataFrame(fps_x_2d.reshape(fps_x_2d.shape[0], shape_hot))  # shape (samples, 20000)
    # return fps_x_1d
    return fps_x_2d

    #
    # # 'blosum62':
    # rows_list = []
    # for seq in list_of_sequences_aa:
    #     protein = descriptors.Descriptor(seq)  # creating object to calculate descriptors)
    #     feature = protein.get_blosum(blosum='blosum62')
    #     rows_list.append(feature)
    # fps_x_blosum = pd.DataFrame(rows_list)
    # rows_list = np.array(rows_list)
    # fps_x_2d = rows_list.reshape(rows_list.shape[0], int(rows_list.shape[1]/23),23)
    #
    # return fps_x_2d
    #
    # elif encoding == 'z-scale':
    # rows_list = []
    # for seq in list_of_sequences_aa:
    #     protein = descriptors.Descriptor(seq)  # creating object to calculate descriptors)
    #     feature = protein.get_z_scales()
    #     rows_list.append(feature)
    # fps_x_zscale = pd.DataFrame(rows_list)
    # rows_list = np.array(rows_list)
    # fps_x_2d = rows_list.reshape(rows_list.shape[0], int(rows_list.shape[1] / 5), 5)
    # return fps_x_2d
    #


t = time.time()
# calculate encodings using multiprocessing
# enconde = Encoding(dataset= dataframe_data,  col= 'seq')
# encode_df = enconde.get_pad_and_hot_encoding(seq_len=21)
# encode_df = enconde.get_hot_encoded()
encode_df = encode_sequence(sequences=dataframe_data['seq'], seq_len=21, padding_truncating='post', encoding='')
# zscale = enconde_df.get_nlf()
# encode_df = enconde.get_blosum()
t2 = time.time()
print('seconds to encoding', t2 - t)
print(encode_df.shape)

# def kfold split but agglomerating sequences similar
groups = dataframe_data['cluster80']  # cluster90  cluster70
# X = encode_df.drop(['seq', 'label', 'type', 'cluster90', 'cluster80', 'cluster70'], axis=1)
# X = encode_df['One_hot_encoding']
y = dataframe_data['label']  # to stratidy based on that

X = encode_df

# DEEP LEARNING
# https://towardsdatascience.com/cross-validate-on-data-containing-groups-correctly-ffa7173a37e6
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
# sgkf.get_n_splits(X=X, y=y, groups=groups)


# DL using o keras Tuner
from basic_models_lstm_keras_tunner import bilstm_builder, bilstm_attention_builder
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', classes= np.unique(y),y = y)
class_weights = {i:w for i,w in enumerate(class_weights)}
input_dim = 21
print(X.shape)

hp = keras_tuner.HyperParameters()
# hypermodel = MyHyperModel()
# bilstm_simple_builder(hp)
# model_builder = bilstm_simple_builder(hp)

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

# model_optimized, best_hps = dl.KerasTuner_get_opt_params(model_builder=bilstm_attention_builder, cv=sgkf, dataX=X, datay=y,
#                                                          scoring=make_scorer(matthews_corrcoef))

# model_optimized = dl.get_model()
# model_optimized = bilstm_simple_builder(best_hps)
# print(model_optimized.summary())
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
scores = dl.cross_validate(bilstm, cv=sgkf,scoring=scoring, return_train_score=False, groups=groups)
print('start cross validate ')
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# scores = dl.train_model_cv(x_cv=X, y_cv=y, cv=sgkf, model=None, groups=groups, refit=True)
print(scores)

# model
# run the model
model = dl.run_model(model=bilstm()) # KerasClassifier(build_fn=bilstm)
# ############################### TESTING SEQUENCES
test_csv = pd.read_csv('/home/martinha/PycharmProjects/viral_fp/viral_fp_new/datasets/test.csv')

for i in range(len(test_csv)):
    seq_fuso = test_csv['Sequence_fusogenic'][i]
    idProtein = test_csv['idProtein'][i]
    name = test_csv['Name'][i]

    # create subsequences
    list_seq, indices = sub_seq_sliding_window(ProteinSequence=seq_fuso,
                                               window_size=21, gap=1)
    # calculates all features
    df = pd.DataFrame()
    df['seq'] = list_seq
    df['indices'] = indices

    # encode_test = Encoding(dataset= dataframe_data ,  col= 'seq')
    # encode_test_df = enconde.get_hot_encoded()
    encode_test_df = encode_sequence(sequences=dataframe_data['seq'], seq_len=21, padding_truncating='post',
                                     encoding='')

    # predict
    predict_df = dl.predict(encode_test_df)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))
