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

seed = 42
np.random.seed(seed)
random.seed(seed)

#get the dataset of the segments

all = 'datasets/all_cluster.csv'
third = 'datasets/third_cluster.csv'
half = 'datasets/half_cluster.csv'

df = pd.read_csv(all)
path = 'results/dl'
report_name = 'all_Window21_gap1_TRANSFORMER_facebook/esm2_t12_35M_UR50D_finetune_class_10groupedKFOLD_cluster80'
# model_name = 'svc'
class_weight = None # 'balanced' # None #
report_path = os.path.join(path, report_name + '/')
if not os.path.exists(report_path):
    os.makedirs(report_path)

REPORT = os.path.join(report_path, report_name)

from sklearn.model_selection import train_test_split

sequences = [str(seq) for seq in df['seq'] ]
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, df['label'], test_size=0.25, shuffle=True)
# https://huggingface.co/facebook/esm2_t6_8M_UR50D
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb#scrollTo=ddbe2b2d
t = time.time()
print(train_sequences)
from transformers import AutoTokenizer
# model_checkpoint = 'facebook/esm2_t6_8M_UR50D' # mais pequeno
model_checkpoint = "facebook/esm2_t12_35M_UR50D"
# model_checkpoint = 'facebook/esm2_t36_3B_UR50D'
# model_checkpoint = 'esm2_t48_15B_UR50D'



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = Dataset.from_dict(test_tokenized)
test_dataset = test_dataset.add_column("labels", test_labels)

from transformers import TFAutoModelForSequenceClassification

num_labels = 2
print("Num labels:", num_labels)
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

tf_train_set = model.prepare_tf_dataset(
    train_dataset,
    batch_size=8,
    shuffle=True,
    tokenizer=tokenizer
)

tf_test_set = model.prepare_tf_dataset(
    test_dataset,
    batch_size=8,
    shuffle=False,
    tokenizer=tokenizer
)
from transformers import AdamWeightDecay

model.compile(optimizer=AdamWeightDecay(2e-5), metrics=["accuracy"])

model.fit(tf_train_set, validation_data=tf_test_set, epochs=3)
model.label2id = {"non_vfp": 0, "vfp": 1}
model.id2label = {val: key for key, val in model.label2id.items()}

model_name = model_checkpoint.split('/')[-1]
finetuned_model_name = f"{model_name}-finetuned-vfp-classification"

# model.push_to_hub(finetuned_model_name)
# tokenizer.push_to_hub(finetuned_model_name)



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
    val_seqs = [str(seq) for seq in list_seq]

    df['indices'] = indices

    val_tokenized = tokenizer(val_seqs)

    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = classifier(val_seqs)
    print(results)

    #
    #
    # results = model(val_tokenized)
    # for result in results:
    #     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    # {'label': 'LABEL_0', 'score': 0.99741131067276},
    # {'label': 'LABEL_0', 'score': 0.9979726672172546},

    df = pd.DataFrame.from_records(results)
    df['class predicted'] = [int(i[6:]) for i in df['label']]

    prob_0 = []
    prob_1 = []

    for i in range(len(df)):
        if df['label'][i] == 'LABEL_0':
            prob_0.append(df['score'][i])
            prob_1.append(1-df['score'][i])
        else:
            prob_1.append(df['score'][i])
            prob_0.append(1-df['score'][i])

    df['prob_class_0'] = prob_0
    df['prob_class_1'] = prob_1

    print(df)
    #
    # val_dataset = Dataset.from_dict(test_tokenized)
    # # tf_val_set = model.prepare_tf_dataset(
    # #     val_dataset,
    # #     batch_size=8,
    # #     shuffle=False,
    # #     tokenizer=tokenizer
    # # )
    #
    # # predict
    # # logits = model(**val_tokenized)
    # tf_predictions = tf.nn.softmax(model(**val_tokenized).logits, axis=-1)
    # print(tf_predictions)
    #
    # predicted_class_id = logits.argmax().item()
    # print(model.config.id2label[predicted_class_id])
    #
    #
    # #
    # # print(model.predict(tf_val_set))
    # # X = model.predict(tf_val_set).to_tuple()
    # # print(X)
    # # print(len(X[0]))
    # # print(len(val_seqs))
    # # print(model.logits)
    # # predictions = np.argmax(X, axis=1)
    # # print(predictions)
    # # # save em files c o nome da seq
    df.to_csv(report_path + 'TESTSEQ{}.csv'.format(idProtein))