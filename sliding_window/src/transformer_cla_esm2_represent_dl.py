import pandas as pd
import numpy as np
import os
import sys
import time
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
os.environ["SM_FRAMEWORK"] = "tf.keras"

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

df = pd.read_csv(third)
path = 'results/dl'
report_name = 'third_Window21_gap1_TRANSFORMER_T33_facebook_esm2_t6_8M_UR50D_token_class_10groupedKFOLD_cluster80_bilstm1'

class_weight = None # 'balanced' # None #
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)

sequences = [str(seq) for seq in df['seq']]

# https://huggingface.co/facebook/esm2_t6_8M_UR50D
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb#scrollTo=ddbe2b2d
t = time.time()
from transformers import AutoTokenizer
model_checkpoint = 'facebook/esm2_t6_8M_UR50D' # mais pequeno
# model_checkpoint = "facebook/esm2_t12_35M_UR50D"
# model_checkpoint = 'facebook/esm2_t36_3B_UR50D'
# model_checkpoint = 'esm2_t48_15B_UR50D'

# pip install fair-esm
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]
# data = [(str(i), sequences[i][:21]) for i in range(len(sequences))] # do not let sequences more than 21 to be easy on lstm shapes otherwise the batchs will give different results on the all dataset
data = [(str(i), sequences[i][:21]) for i in range(len(sequences))] # do not let sequences more than 21 to be easy on lstm shapes otherwise the batchs will give different results on the all dataset


# seq_all = []
# token_all = []
# contact_all = []
# b_size = 1500
# for index in range(0, len(data), b_size):
#     representatcdions = {}
#     if index + b_size <= len(data):
#         seq = data[index: index + b_size]
#     else:
#         seq = data[index: len(data)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]
#     results = model(batch_tokens, repr_layers=[6], return_contacts=True)
# token_representations = results["representations"][6]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

sequence_representations = np.array([np.array(seq) for seq in sequence_representations])
print(sequence_representations.shape) #[413,1280]
print(token_representations.shape) #[413,49,1280]

contact_representations = results["contacts"]
print(contact_representations.shape)    # [413,47,47]

    # seq_all.extend(sequence_representations)
    # contact_all.extend(contact_representations)
    # token_all.extend(token_representations)


X = np.array(token_representations)
# X = np.array(seq_all)
# X = np.array(contact_representations)


# def kfold split but agglomerating sequences similar
groups = df['cluster80']  # cluster90
y = df['label']  # to stratidy based on that
# DL using o keras Tuner
from basic_models_lstm_keras_tunner import bilstm_builder, bilstm_attention_builder
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', classes= np.unique(y),y = y)
class_weights = {i:w for i,w in enumerate(class_weights)}
input_dim = 47
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

# scores = dl.cross_validate(bilstm, cv=sgkf,scoring=scoring, return_train_score=False, groups=groups)
# print(scores)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# scores = dl.train_model_cv(x_cv=X, y_cv=y, cv=sgkf, model=None, groups=groups, refit=True)

# model
# run the model
model_dl = dl.run_model(model=bilstm()) # KerasClassifier(build_fn=bilstm)

############################### TESTING SEQUENCES
test_csv = pd.read_csv('/ViralFP_dataset/data/holdout_dataset.csv')

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

    data_test = [(str(i), list_seq[i]) for i in range(len(list_seq))]

    batch_labels, batch_strs, batch_tokens = batch_converter(data_test)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    #     results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    # token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))


    sequence_representations = np.array([np.array(seq) for seq in sequence_representations])
    print(sequence_representations.shape) #[413,1280]
    print(token_representations.shape) #[413,49,1280]

    contact_representations = results["contacts"]
    print(contact_representations.shape)    # [413,47,47]

    X_test = np.array(token_representations)
    # X_test = np.array(contact_representations)

    # predict
    predict_df = dl.predict(X_test)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))
