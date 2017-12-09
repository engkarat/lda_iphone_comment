import numpy as np


# hyperparams
k = 5

def lda(np_data, k):
    n_row, n_col = np_data.shape
    np_topic = np.random.rand(n_row, n_col)
    dkm, kwm = construct_matrix(np_topic, k)
    print()
    return

def construct_matrix(np_data, k):
    n_row, n_col = np_data.shape
    dkm = np.zeros([n_row, k])
    kwm = np.zeros([n_col, k])
    for n in range(n_row):
        row_group, row_freq = np.unique(np_data[n, :], return_counts=True)
        dkm[n, :] = row_freq
    for n in range(n_col):
        col_group, col_freq = np.unique(np_data[:, n], return_counts=True)
        kwm[n, :] = col_freq
    return dkm, kwm

if __name__=="__main__":
    np_data = np.genfromtxt('out_file/test_set.csv', delimiter=',')
    lda(np_data, k)