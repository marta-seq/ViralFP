import pandas as pd
import numpy as np
import os
import sys
import time
import random

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('/home/martinha/PycharmProjects/viral_fp/')

# from propythia_new.src.propythia.sequence import ReadSequence
# from propythia_new.src.propythia.protein_descriptors import ProteinDescriptors
# from propythia_new.src.propythia.preprocess import Preprocess
# from propythia_new.src.propythia.shallow_ml import ShallowML
# from propythia_new.src.propythia.sequence import sub_seq_sliding_window
# from propythia_new.src.propythia.feature_selection import FeatureSelection

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
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
sys.path.append('/home/martinha/PycharmProjects/protein/viral_fp/')

# from propythia_new.src.propythia.sequence import ReadSequence
# from propythia_new.src.propythia.protein_encoding import Encoding
# from propythia_new.src.propythia.deep_ml import DeepML

# from propythia_new.src.propythia.sequence import sub_seq_sliding_window

from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

# from datasets.IO import fasta_to_dict, read_cluster_file
from tensorflow.keras.utils import to_categorical

seed = 42
np.random.seed(seed)
random.seed(seed)

file_train = pd.read_csv(
    '/home/martinha/PycharmProjects/protein/viral_fp/viral_fp_new/datasets/viral_fp_curated_without_test.csv')

path = '/home/martinha/PycharmProjects/protein/viral_fp/viral_fp_new/src/classify_full_seq/results/dl'
report_name = 'seq_transformers_token_10KFOLD_cluster80_epoch5_T12_try2'
class_weight = None  # 'balanced' # None #
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)
EPOCHS = 5
# file_train drop nas in sequence fusogenic and peptide
df_train = file_train.dropna(subset=['Sequence_fusogenic', 'seq'])

# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb#scrollTo=78d701ed
# nstead of classifying the whole sequence into a single category, we categorize each token (amino acid, in this case!) into one or more categories.
#
# In this section, we're going to gather some training data from UniProt.' \
# As in the sequence classification example, we aim to create ' \
#                    'two lists: sequences and labels. Unlike in that example, however, t
# he labels are more than just single integers. Instead, the label for each sample will be one integer per token in the input. This should make sense - when we do token classification, different tokens in the input may have different categories!

# get the y  000 1111 0000 where the 1s are the fsion peptide
# Y transform sequence
y = []
index_list = []
df_train = df_train.reset_index()
for i in range(len(df_train)):
    fusion_protein = df_train['Sequence_fusogenic'][i]
    fusion_peptide = df_train['seq'][i]
    replace_x = '1' * len(fusion_peptide)
    new_protein_x = str(fusion_protein).replace(str(fusion_peptide), replace_x)
    index = new_protein_x.find(replace_x)
    protein_y = [1 if x == '1' else 0 for x in new_protein_x]
    # print(protein_y)
    # print(len(protein_y))
    y.append(protein_y)
    index_list.append(index)

print('#########################################X####################')
sequences = [str(seq[:1500]) for seq in df_train['Sequence_fusogenic']]
from sklearn.model_selection import train_test_split

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, y, test_size=0.25,
                                                                              shuffle=True)
from transformers import AutoTokenizer

# model_checkpoint = 'facebook/esm2_t6_8M_UR50D'  # + pequeno
model_checkpoint = 'facebook/esm2_t12_35M_UR50D'
# model_checkpoint = 'facebook/esm2_t36_3B_UR50D'
# esm2_t48_15B_UR50D()


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

from datasets import Dataset

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

# The key difference here with the above example is that we use TFAutoModelForTokenClassification
# instead of TFAutoModelForSequenceClassification. We will also need a data_collator this time,
# as we're in the slightly more complex case where both inputs and labels must be padded in each batch.

from transformers import TFAutoModelForTokenClassification

num_labels = 2
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="np")

# Now we create our tf.data.Dataset objects as before. Remember to pass the data collator, though!
# Note that when you pass a data collator, there's no need to pass your tokenizer, as the data collator
# is handling padding for us.


tf_train_set = model.prepare_tf_dataset(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator
)

tf_test_set = model.prepare_tf_dataset(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator
)

# Our metrics are bit more complex than in the sequence classification task, as we need to ignore padding
# tokens (those where the label is -100). This means we'll need our own metric function where we only compute accuracy on non-padding tokens.

from transformers import AdamWeightDecay
import tensorflow as tf


def masked_accuracy(y_true, y_pred):
    predictions = tf.math.argmax(y_pred, axis=-1)  # Highest logit corresponds to predicted category
    numerator = tf.math.count_nonzero((predictions == tf.cast(y_true, predictions.dtype)) & (y_true != -100),
                                      dtype=tf.float32)
    denominator = tf.math.count_nonzero(y_true != -100, dtype=tf.float32)
    return numerator / denominator


model.compile(optimizer=AdamWeightDecay(2e-5), metrics=[masked_accuracy])

model.fit(tf_train_set, validation_data=tf_test_set, epochs=EPOCHS)

model.label2id = {"non_vfp": 0, "vfp": 1}
model.id2label = {val: key for key, val in model.label2id.items()}

model_name = model_checkpoint.split('/')[-1]
finetuned_model_name = f"{model_name}-finetuned-secondary-structure-classification"

print('finished')

#
# This definitely seems harder than the first task, but we still attain a very respectable
# accuracy. Remember that to keep this demo lightweight, we used one of the smallest ESM models,
# focused on human proteins only and didn't put a lot of work into making sure we only included
# completely-annotated proteins in our training set.
# With a bigger model and a cleaner, broader training set, accuracy on this task could definitely
# go a lot higher!


# testar c modelo maior


############################### TESTING SEQUENCES
test_csv = pd.read_csv('/home/martinha/PycharmProjects/protein/viral_fp/viral_fp_new/datasets/test.csv')

for i in range(len(test_csv)):
    seq_fuso = test_csv['Sequence_fusogenic'][i]
    idProtein = test_csv['idProtein'][i]
    name = test_csv['Name'][i]


    from transformers import pipeline
    classifier = pipeline("ner", model=model, tokenizer=tokenizer)
    results = classifier(seq_fuso)

    # {'entity': 'LABEL_0', 'score': 0.9992617, 'index': 594, 'word': 'D', 'start': None, 'end': None},
    # {'entity': 'LABEL_0', 'score': 0.99876815, 'index': 595, 'word': 'D', 'start': None, 'end': None}

    df = pd.DataFrame.from_records(results)
    df['class predicted'] = [int(i[6:]) for i in df['entity']]

    prob_0 = []
    prob_1 = []

    for i in range(len(df)):
        if df['entity'][i] == 'LABEL_0':
            prob_0.append(df['score'][i])
            prob_1.append(1-df['score'][i])
        else:
            prob_1.append(df['score'][i])
            prob_0.append(1-df['score'][i])

    df['prob_class_0'] = prob_0
    df['prob_class_1'] = prob_1

    print(df)

    df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))


# https://github.com/facebookresearch/esm
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb#scrollTo=3f683dd7
# https://huggingface.co/facebook/esm1b_t33_650M_UR50S
# https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#pipeline
# https://github.com/facebookresearch/esm