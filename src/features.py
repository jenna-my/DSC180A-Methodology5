import pandas as pd

def return_int(x):
    return [int(i) for i in x.split(';')[:-1]]
# Find longest sequence length
def longest_seq(aList):
    maxCount = 1
    actualCount = 1
    for i in range(len(aList)-1):
        if aList[i] == aList[i+1]:
            actualCount += 1
        else:
            actualCount = 1
        if actualCount > maxCount:
            maxCount = actualCount
    return maxCount
    
def apply_features(df, outfile = None):
    # Find max bytes per second
    df['max_bytes'] = df.groupby('Second')['Total Bytes'].transform('max')
    # Find longest sequence
    df['packet_dirs'] = df['packet_dirs'].astype('str').apply(return_int)
    df['longest_seq'] = df['packet_dirs'].apply(longest_seq)

    # Calculate features for loss- std of total bytes, longest seq, and var of max bytes
    df['std_bytes_loss'] = df.groupby("loss")['Total Bytes'].transform('std')
    df['var_longest_seq_loss'] = df.groupby("loss")['longest_seq'].transform('var')
    df['var_max_bytes_loss'] = df.groupby("loss")['max_bytes'].transform('var')

#     # Calculate features for latency- std of total bytes, longest seq, and var of max bytes
#     df['std_bytes_lat'] = df.groupby("latency")['Total Bytes'].transform('std')
#     df['var_longest_seq_lat'] = df.groupby("latency")['longest_seq'].transform('var')
#     df['var_max_bytes_lat'] = df.groupby("latency")['max_bytes'].transform('var')
    
    features = df[['std_bytes_loss', 'var_longest_seq_loss', 'Total Bytes']]
    labels = df['loss']
    if outfile:
        features.to_csv(outfile, index=False)
    return features, labels