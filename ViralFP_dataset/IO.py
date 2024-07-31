import pandas as pd
from Bio import SeqIO
import re

def make_fasta(df, filename_fasta,seq_column):
    with open(filename_fasta, 'w') as out:
        for index, row in df.iterrows():
            id = row['UniProtID']
            seq = row[seq_column]
            name = row['Name']
            clas = row['Class']
            # label = row['label']
            out.write('>' + str(id) + '_' + str(name) + '_' + str(clas) + '_' + str(index)
                      + '\n' + seq.strip() + '\n')


def read_fasta(fasta90, th = 90):
    with open(fasta90) as fasta_file:  # Will close handle cleanly
        seq = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            seq.append(seq_record.seq)
    print('number of sequences with {} % threshold: {}'. format(th, len(seq)))
    return seq


def fasta_to_dict(fasta_or = 'fasta_curated.fasta'):
    # create a dict with name of sequence as key and sequence as value.
    # get a dict with sequences and name
    dict_or = dict()
    with open(fasta_or) as fasta_file:
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            dict_or[seq_record.id] = seq_record.seq
    return dict_or


def read_cluster_file(cl_file, dict_or):
    # dict _or tem so keys os names no fasta e value sa sequnce
    dict_cluster = dict()
    key_list = []
    cluster_name = 'None'
    with open(cl_file) as cluster_file:
        for line in cluster_file:

            if line.startswith('>'):
                dict_cluster[cluster_name] = key_list # from previous
                # reset the name and the list
                cluster_name = line # Cluster 0
                key_list = []

            else:
                m = re.search('>(.+?)\\.', line)
                name = m.group(1)
                #                 name = 'between Z and ...' #>nan_HA_I_247...
                key_list.append(dict_or[name])
    dict_cluster.pop('None')
    return dict_cluster
