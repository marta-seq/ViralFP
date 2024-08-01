import pandas as pd
import sys
import os
import time
import random
import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


####

# function from ProPythia to obtain subsequences in window approach
def sub_seq_sliding_window(ProteinSequence, window_size=20, gap=1, index=True):
    """
    sliding window of the protein given. It will generate a list of n sequences with
    lenght equal to the value of window and spaced a gap value. It can or not retrieve
    the indices of location of the subsequence in the original sequence.

    :param ProteinSequence: protein sequence
    :param window_size: number of aminoacids to considerer, lenght of the subsequence. for default 20
    :param gap: gap size of the search of windows in sequence. default 1
    :param index: if true, return the indices of location of the subsequence in the original sequence
    :return: list with subsequences generated with or without a list of tuples with location of subsequences
     in original sequence
    """

    m = len(ProteinSequence)
    n = int(window_size)
    list_of_sequences = []
    indices = []

    for i in range(0, m - (n - 1), gap):
        list_of_sequences.append(ProteinSequence[i:i + n])
        indices.append((i, i + n))
        i += 1
        if len(ProteinSequence[i:i + n]) == 0: print(ProteinSequence)
    if index:
        return list_of_sequences, indices
    else:
        return list_of_sequences

def get_maximum_scores_subsequence(scores, sequence):
    # Length of the subsequences
    subseq_len = 21

    # Initialize a list of predictions with zeros
    predictions = [0] * len(sequence)

    # Iterate over the scores and update the predictions list for each subsequence
    for i, score in enumerate(scores):
        start_idx = i * (subseq_len - 1)
        end_idx = min(start_idx + subseq_len, len(sequence))
        for j in range(start_idx, end_idx):
            predictions[j] = max(predictions[j], score)

    return predictions
    # Print the predictions
    # print('predictions', len(predictions))
    # print(len(sequence))

def make_bin_sequences_and_jaccard(original_sequence, subsequence, predictions):
    # binarize the original sequence
    # Create a list of 0s with the same length as the original sequence
    # true_bin = [0.0] * len(original_sequence)
    # # Iterate through each subsequence in the list
    # for subseq in subsequence:
    #     # Identify the starting position of the subsequence within the original sequence
    #     start_pos = original_sequence.find(subseq)
    #     # Replace the corresponding positions in the list of 0s with 1s, representing the subsequence
    #     true_bin[start_pos:start_pos+len(subseq)] = [1.0]*len(subseq)


    # # Create a list of 0s with the same length as the original sequence
    print(subsequence)
    for sub in subsequence:
        true_bin = [0.0] * len(original_sequence)
        # Identify the starting position of the subsequence within the original sequence
        start_pos = sequence.find(sub)
        # Replace the corresponding positions in the list of 0s with 1s, representing the subsequence
        true_bin[start_pos:start_pos+len(sub)] = [1.0]*len(sub)

    # predictions into 0 and 1

    # # use directly the scores from subsequences prob_class_1
    # scores_bin = [1 if x >= threshold else 0 for x in predictions]
    # # print(len(scores_bin))
    # # print(subsequence)
    #
    # scores_bin += [scores_bin[-1]] * 20     # add the 20 final aas
    # has a problem
    # 0,1,0.36,0.64
    # 1,1,0.35,0.65
    # 2,0,0.51,0.49
    # 3,0,0.5,0.5
    # 4,0,0.63,0.37
    # 5,0,0.62,0.38
    # 6,0,0.73,0.27

    #
    # # use max scores
    # max_scores = get_maximum_scores_subsequence(scores, sequence)
    # max_scores_bin = [1 if x >= threshold else 0 for x in max_scores]
    scores_bin = [1 if x >= threshold else 0 for x in predictions]

    jaccard_index = jaccard_score(y_true = true_bin, y_pred = scores_bin)

    jaccard_index_max = jaccard_score(y_true = true_bin, y_pred = scores_bin)
    return jaccard_index, jaccard_index_max




def get_subseqs_original_subseq(sequence, subsequence):
    # get a list of subsequences with scores of positive or negative
    # the border ones 3 should be deleted ?
    for i in range(len(subsequence)):
        subseq = subsequence[i]
        start_idx = sequence.index(subseq)-1
        end_idx = start_idx + len(subseq)
        border_tolerance = 3
        replace_x = 'X' * len(subseq)

        before_subseq = max(start_idx - 3, 0)
        after_subseq = min(end_idx + 5, len(sequence))
        new_seq = sequence[:before_subseq] + "X" * (start_idx - before_subseq) + replace_x
        new_seq += "X" * (after_subseq - end_idx)
        new_seq += sequence[after_subseq:]
        sequence = new_seq

    list_seq, indices = sub_seq_sliding_window(ProteinSequence=sequence,
                                               window_size=21, gap=1)
    # Create an empty dataframe
    df = pd.DataFrame()
    # Add the sequences to the dataframe as a new column
    df['sequence'] = list_seq
    # Define a lambda function to calculate the percentage of 'X' in the sequence
    x_percent = lambda seq: (seq.count('X') / len(seq)) if 'X' in seq else None

    # Add a new column to the dataframe with the percentage of 'X' in the sequence
    df['x_percent'] = df['sequence'].apply(x_percent)
    df['true'] = df['x_percent'].fillna(0.0)
    return df


def get_seqs_predicted(df, sequence):
    subsequences = []
    # find the index where predictions are 1
    index_list = df.loc[df['y_pred'] == 1].index.tolist()
    print(index_list)
    # initialize the output list
    output_list = []
    if index_list:
        # loop through the index list and group consecutive indices together
        start = end = index_list[0]
        for i in index_list[1:] + [None]:
            if i != end + 1:
                output_list.append((int(start), int(end)))
                start = i
            end = i
        for i in output_list:
            subsequences.append(sequence[i[0]: i[1]+1])

    return subsequences




if __name__ == '__main__':
    filenames_test_results = [105, 146,158, 192,225,265, 574,618,779,804]
    test_csv = pd.read_csv('ViralFP_dataset/data/holdout_dataset.csv')

    threshold = 0.5
    modelid_path = 'results/'
    modelid = modelid_path + 'seq_transformers_token_10KFOLD_cluster80_epoch3_T12_try2/'
    for seq_test in filenames_test_results:
        filename = modelid + 'TESTSEQ' + str(seq_test) + '.csv'
        sequence = test_csv.loc[test_csv['idProtein']==seq_test, 'Sequence_fusogenic'].iloc[0]
        subsequence = test_csv.loc[test_csv['idProtein']==seq_test,'seq_vfp'].iloc[0]
        scores = pd.read_csv(filename)

        scores = scores['prob_class_1']

        # subsequence = subsequence.split(' ')
        # print(subsequence)
        # subsequence = list(subsequence)
        if '[FRWYGPKY' in subsequence:
            subsequence = ['FRWYGPKY', 'CGYATVT']
        else:
            subsequence = [subsequence]
        true_bin = [0.0] * len(sequence)
        for sub in subsequence:
            # Identify the starting position of the subsequence within the original sequence
            start_pos = sequence.find(sub)
            # Replace the corresponding positions in the list of 0s with 1s, representing the subsequence
            true_bin[start_pos:start_pos+len(sub)] = [1.0]*len(sub)

        scores_bin = [1 if x >= threshold else 0 for x in scores]

        jaccard_index = jaccard_score(y_true = true_bin, y_pred = scores_bin)
        jaccard_index_max = jaccard_score(y_true = true_bin, y_pred = scores_bin)

        y_true = true_bin
        y_pred = scores_bin
        df=pd.DataFrame([])
        df['y_pred'] = y_pred

        # # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        seqs_pred = get_seqs_predicted(df, sequence)




        # Open a file for writing
        filename_scores = modelid + 'TESTSEQ' + str(seq_test) + 'scores.txt'
        csv_scores = modelid + 'TESTSEQ' + str(seq_test) + 'scores.csv'
        with open(filename_scores, 'w') as f:
            # Redirect standard output to the file
            sys.stdout = f
            df.to_csv(csv_scores, index=False)

            print(subsequence)
            print(df)
            # Display the metrics
            print('Accuracy:', accuracy)
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1:', f1)
            print('ROC AUC:', roc_auc)
            print('MCC:', mcc)
            print("Jaccard index:", jaccard_index)
            print("Jaccard index max scores:", jaccard_index)

            print('sequences predicted')
            print(seqs_pred)
            # Reset standard output to the console
            sys.stdout = sys.__stdout__
            print(filename_scores)
        print(subsequence)
        print(df)
        # Display the metrics
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('ROC AUC:', roc_auc)
        print('MCC:', mcc)
        print("Jaccard index:", jaccard_index)
        print("Jaccard index max scores:", jaccard_index)
        print('sequences predicted')
        print(seqs_pred)

        print(filename_scores)