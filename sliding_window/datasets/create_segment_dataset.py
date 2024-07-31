import copy

import pandas as pd
import os
import sys
# sys.path.append('../src/datasets')
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import random
from IO import fasta_to_dict


random.seed(42)

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


def create_segments(fusion_protein_list, fusion_peptide_list, window_size=21, gap=1, border_tolerance=3):
    if window_size is not str:
        window = window_size
    else:
        window = None  # will be the len of fusion peptide of that line

    list_seq_neg = []
    list_seq_pos = []

    for i in range(len(fusion_peptide_list)):
        fusion_protein = fusion_protein_list[i]
        fusion_peptide = fusion_peptide_list[i]
        if window_size is None: window = len(fusion_peptide)

        if fusion_protein:
            replace_x = 'X' * len(fusion_peptide)
            new_protein_x = str(fusion_protein).replace(str(fusion_peptide), replace_x)
            index = new_protein_x.find(replace_x)

            # add some extra X to not count as negatives as the borders of fusion peptides are not well established.
            l = list(new_protein_x)
            l[index - border_tolerance:index] = 'X' * border_tolerance
            l[index + len(fusion_peptide): index + border_tolerance] = 'X' * border_tolerance
            new_protein_x = ''.join(l)
            list_seq, indices = sub_seq_sliding_window(ProteinSequence=new_protein_x,
                                                       window_size=window, gap=gap)
            ls = [x for x in list_seq if "X" not in x]
            list_seq_neg.append(ls)
        else:
            list_seq_neg.append([])

        list_seq_pos.append([fusion_peptide])
    return list_seq_pos, list_seq_neg

#
# # delete all negative from neg pool subsequence that have 70 % similarity with fusion peptide
# delete all negative from neg pool transmembrane that have 70 % similarity with fusion peptide

# D-HIT-2D compares 2 protein datasets (db1, db2).
# It identifies the sequences in db2 that are similar to db1 at a certain threshold.
# The input are two protein datasets (db1, db2) in fasta format
# and the output are two files: a fasta file of proteins in db2 that are not similar to db1
# and a text file that lists similar sequences between db1 & db2.

# create fasta from lists
def make_fasta_from_list(list_protein, filename_fasta):
    with open(filename_fasta, 'w') as out:
        for i in range(len(list_protein)):
            for idsub in range(len(list_protein[i])):
                id = 'seq{}_subseq{}'.format(i, idsub)
                out.write('>' + str(id)
                          + '\n' + list_protein[i][idsub].strip() + '\n')


file_train = pd.read_csv(
    '../ViralFP_dataset/data/viral_fp_curated_without_test.csv')

list_seq_pos, list_seq_neg = create_segments(
    fusion_protein_list=file_train['Sequence_fusogenic'], fusion_peptide_list=file_train['seq_vfp'],
    window_size=21,
    gap=1,
    border_tolerance=3)

make_fasta_from_list(list_protein=list_seq_neg, filename_fasta='df_subseq_neg')
make_fasta_from_list(list_protein=list_seq_pos, filename_fasta='df_pos')

# eliminate all sequences in subsequences and in tmd that have similarity with vfp
os.system('cd-hit-2d -i df_pos -i2 df_subseq_neg -o df_subseq_neg_new07 -c 0.7 -n 5 -d 0 -M 16000 -T 8 -s2 0.9')
os.system(
    'cd-hit-2d -i df_pos -i2 fasta_transmembrane_domains.txt -o df_tmd_neg07 -c 0.7 -n 5 -d 0 -M 16000 -T 8 -s2 0.9')

# # eliminate tmds that have similarities with subsequences
os.system('cd-hit-2d -i df_subseq_neg_new07 -i2 df_tmd_neg07 -o df_tmd_neg_new2 -c 0.7 -n 5 -d 0 -M 16000 -T 8 -s2 0.9')

# take out sequences equal inside tmd and inside
os.system('cd-hit -i df_subseq_neg_new07 -o subseq_neg07 -c 0.7 -n 5')
os.system('cd-hit -i df_tmd_neg_new2 -o tmd_neg07 -c 0.7 -n 5')

# similarity between the pos
os.system('cd-hit -i df_pos -o df_pos08 -c 0.8 -n 5')
os.system('cd-hit -i df_pos -o df_pos09 -c 0.9 -n 5')
os.system('cd-hit -i df_pos -o df_pos07 -c 0.7 -n 5')

# count values of the final pools
#
#
# dict_subseq_neg = fasta_to_dict(fasta_or='subseq_neg07')
# dict_tmd_neg = fasta_to_dict(fasta_or='tmd_neg07')


dict_pos_08 = fasta_to_dict(fasta_or='df_pos08')
dict_pos_09 = fasta_to_dict(fasta_or='df_pos09')
dict_pos = fasta_to_dict(fasta_or='df_pos')

print('subsequences negatives', len(dict_subseq_neg))
print('tmd negatives', len(dict_tmd_neg))
print('sequences vfp', len(list_seq_pos))
print('sequences vfp', len(file_train['seq_vfp'].drop_duplicates()))
print('sequences vfp with less than 90 similarity', len(dict_pos_09))
print('sequences vfp with less than 80 similarity', len(dict_pos_08))

#
# # make datasets with different ratios
#
# # subsequences negatives 5819
# # tmd negatives 784
# # sequences vfp 394
# # sequences vfp with less than 90 similarity 112
# # sequences vfp with less than 80 similarity 83


# datasets will take always the positive (even duplicates - then will be deleted and agglomerated)
df = pd.DataFrame()

# positives 207 (non duplicates vfp)
n_pos = [1] * len(file_train['seq_vfp'].drop_duplicates())

# all negatives
tmd_values = [str(seq) for seq in list(dict_tmd_neg.values())]
subseq_values = [str(seq) for seq in list(dict_subseq_neg.values())]
pos_values = list(file_train['seq_vfp'].drop_duplicates())

df_all =pd.DataFrame()
list_all = pos_values + subseq_values + tmd_values
n_neg = [0] * (len(list_all) - len(n_pos))
df_all['seq'] = list_all
df_all['label'] = n_pos + n_neg
df_all['type'] = ['vfp'] * len(n_pos) + \
                 ['subseq'] * len(dict_subseq_neg.values()) + \
                 ['tmd'] * len(dict_tmd_neg.values())

print(df_all)
print('all', df_all.shape)
print('positives', df_all.loc[df_all['label']==1].shape)
print('negative', df_all.loc[df_all['label']==0].shape)
print('saving')
df_all.to_csv(
    '/home/martinha/PycharmProjects/viral_fp/viral_fp_new/src/classify_segments/datasets/all_negatives_.csv', index=False)


# get list the subsequences from each entrie
list_seq_neg_filtered_fasta = []
subseq_neg_new = copy.deepcopy(subseq_values)
for list_seq in list_seq_neg:
    new_list_seq= []
    for seq in list_seq:
        if seq in subseq_neg_new:
            new_list_seq.append(seq)
            subseq_neg_new.remove(seq) # because it may have repetitions
    list_seq_neg_filtered_fasta.append(new_list_seq)

print(len(list_seq_neg_filtered_fasta)) # 394 entries
#
# # df quarter and third
# # positives are the same
# # one subseq per vfp w or w/o duplicate entries of vfp
# # 207 random tmd
#
df = copy.deepcopy(df_all)
df_pos = df_all.loc[df_all['label']==1] # no duplicates
df_tmd = df_all.loc[df_all['type'] == 'tmd'].sample(len(n_pos))

df_subseq_list = [random.choice(seq) for seq in list_seq_neg_filtered_fasta if len(seq)>1] # one element per line
print('===========')
print(len(df_subseq_list))
#
# df['dup_vfp'] = list_seq_pos
# df['subseq'] = df_subseq
#
# df = df.drop_duplicates(subset = ['dup_vfp']) # remove the subseqs that belong to duplicate entries
# df_subseq_no_dup = df['subseq']
#
# print(len(df_subseq_no_dup))
#
# df 25 pos 50 subseq 25 tmd
df_subseq = pd.DataFrame()
df_all['seq'] = df_subseq
df_all['label'] = [0] * len(df_subseq)
df_all['type'] = ['subseq'] * len(df_subseq)


df_quarter = df_pos + df_tmd + df_subseq

print(df_quarter)
print('all', df_quarter.shape)
print('positives', df_quarter.loc[df_quarter['label']==1].shape)
print('negative', df_quarter.loc[df_quarter['label']==0].shape)
print('saving')
df_quarter.to_csv(
    '/home/martinha/PycharmProjects/viral_fp/viral_fp_new/src/classify_segments/datasets/quarter.csv', index=False)


# df 33 33 33 subseq only from no ndup ( are the same from the quarter)
df_subseq = pd.DataFrame()
df_subseq['seq'] = df_subseq_list
df_subseq['label'] = [0] * len(df_subseq_list)
df_subseq['type'] = ['subseq'] * len(df_subseq_list)

df_third = df_pos.append(df_tmd).append(df_subseq)


# print(df_third)
print('all', df_third.shape)
print('positives', df_third.loc[df_third['label']==1].shape)
print('negative', df_third.loc[df_third['label']==0].shape)
print('saving')
df_third.to_csv(
    '/home/martinha/PycharmProjects/viral_fp/viral_fp_new/src/classify_segments/datasets/third.csv', index=False)

# df 50 25 25 subseq only from no ndup ( are the same from the quarter)

df_tmd_half = df_tmd.sample(len(n_pos)//2)
df_subseq_half = df_subseq.sample(len(n_pos)//2)
df_half = df_pos.append(df_tmd_half).append(df_subseq_half)

# print(df_third)
print('all', df_half.shape)
print('positives', df_half.loc[df_half['label']==1].shape)
print('negative', df_half.loc[df_half['label']==0].shape)
print('saving')

df_half.to_csv(
    '/home/martinha/PycharmProjects/viral_fp/viral_fp_new/src/classify_segments/datasets/half.csv', index=False)

#
# # df_all
# # all (6810, 3)
# # positives (207, 3)
# # negative (6603, 3) # 5819 subseq 784 tmd
#
#
# # df_third
# # all (645, 3)
# # positives (207, 3)
# # negative (438, 3)  # 231 subseq 207 tmd
#
#
# # df_half
# # all (413, 3)
# # positives (207, 3)
# # negative (206, 3) # 103 subseq 103 tmd
#
#
# # get the clusters of the positives
#
import os

import pandas as pd

os.system('cd-hit -i df_pos -o df_pos08 -c 0.8 -n 5')
os.system('cd-hit -i df_pos -o df_pos09 -c 0.9 -n 5')
os.system('cd-hit -i df_pos -o df_pos07 -c 0.7 -n 5')

# get the clusters from the vfp
from IO import fasta_to_dict, read_cluster_file


dict_or = fasta_to_dict(fasta_or = 'df_pos')
dict_cl90 = read_cluster_file('df_pos09.clstr', dict_or)
dict_cl80 = read_cluster_file('df_pos08.clstr', dict_or)
dict_cl70 = read_cluster_file('df_pos07.clstr', dict_or)

def get_cl_goups(dict_cl90, df):
    seq_list = []
    cluster = []
    n = 0
    df_cluster = pd.DataFrame()
    for key in dict_cl90.keys():
        values = dict_cl90[key]
        for seq in values:
            seq_list.append(str(seq))
            cluster.append(n)
        n+=1
    df_cluster['seq_list'] = seq_list
    df_cluster['cluster'] = cluster
    df_cluster = df_cluster.drop_duplicates()

    cluster = []
    n = 1000
    for seq in df['seq']:
        if seq in seq_list:
            cl = df_cluster.loc[df_cluster['seq_list']==seq, 'cluster'].to_numpy()[0]
            cluster.append(cl)
        else:
            cluster.append(n)
            n+=1
    return cluster

df_half = (
    pd.read_csv('datasets/mldatasets/half.csv'))

df_half['cluster90'] = get_cl_goups(dict_cl90, df_half)
df_half['cluster80'] = get_cl_goups(dict_cl80, df_half)
df_half['cluster70'] = get_cl_goups(dict_cl70, df_half)
df_half.to_csv(
    'datasets/mldatasets/half_cluster.csv', index=False)

df_third = pd.read_csv('datasets/mldatasets/third.csv')
df_third['cluster90'] = get_cl_goups(dict_cl90, df_third)
df_third['cluster80'] = get_cl_goups(dict_cl80, df_third)
df_third['cluster70'] = get_cl_goups(dict_cl70, df_third)
df_third.to_csv(
    'datasets/mldatasets/third_cluster.csv', index=False)

df_all = pd.read_csv('datasets/mldatasets/all_negatives_.csv')
df_all['cluster90'] = get_cl_goups(dict_cl90, df_all)
df_all['cluster80'] = get_cl_goups(dict_cl80, df_all)
df_all['cluster70'] = get_cl_goups(dict_cl70, df_all)
df_all.to_csv(
    'datasets/mldatasets/all_cluster.csv', index=False)

print(df_all)