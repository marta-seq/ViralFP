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

from propythia.shallow_ml import ShallowML
from propythia.sequence import sub_seq_sliding_window

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
seed = 42
np.random.seed(seed)
random.seed(seed)

#get the dataset of the segments

all = 'datasets/all_cluster.csv'
third = 'datasets/third_cluster.csv'
half = 'datasets/half_cluster.csv'

df = pd.read_csv(all)
path = 'results/ml'
report_name = 'all_Window21_gap1_TRANSFORMER_facebook_esm2_t6_8M_UR50D_class_class_10groupedKFOLD_cluster80_gnb'
model_name = 'gnb'
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
data = [(str(i), sequences[i]) for i in range(len(sequences))]

seq_all = []
token_all = []
contact_all = []
b_size = 1500
for index in range(0, len(data), b_size):
    representations = {}
    if index + b_size <= len(data):
        seq = data[index: index + b_size]
    else:
        seq = data[index: len(data)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq)
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

    seq_all.extend(sequence_representations)
    contact_all.extend(contact_representations)
    token_all.extend(token_representations)


# X = sequence_representations
X = np.array(seq_all)
# Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(seq)
#     plt.show()



# def kfold split but agglomerating sequences similar
groups = df['cluster80']  # cluster90
y = df['label']  # to stratidy based on that

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

    X_test = sequence_representations

    # predict
    predict_df = ml.predict(X_test)

    # save em files c o nome da seq
    predict_df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))



    # https://github.com/facebookresearch/esm

