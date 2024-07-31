import pandas as pd


path_file = 'data/viral_fp_original_data.csv'

file = pd.read_csv(path_file)

# Corrected some VFP for DL but others are only in seq
seq_vfp = []
for x in range(len(file)):
    if not pd.isna(file['FP corrected DL'][x]):
        print('x')
        seq_vfp.append(file['FP corrected DL'][x])
    else:
        seq_vfp.append(file['seq'][x])

file['seq_vfp'] = seq_vfp



#
# remove 124 125 127 . 124 and 125 dont have fusion protein and the fusion peptide is the same as 126
# (ending before that the one from 126)
# 127 has the same protein fusion protein and fusion peptide and is a smaller version of the one from 126.





# remove duplicates with seq fusogenic and peptide equal and drop any peptide with Nan
file_no_du = file.dropna(subset=['seq_vfp']) # 411
file_no_du = file.drop_duplicates(subset=['seq_vfp','Sequence_fusogenic'], keep=False)


file_no_du = file_no_du.dropna(subset=['Sequence_fusogenic']) # 379
