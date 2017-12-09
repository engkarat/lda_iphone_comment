import numpy as np


# hyperparams
k = 5

def lda(np_data, k):
    n_row, n_col = np_data.shape
    np_topic = np.random.rand(n_row, n_col)*k
    np_topic = np_topic.astype(np.int)
    dkm, kwm = construct_matrix(np_data, np_topic, k)
    print(dkm)
    return

def construct_matrix(np_data, np_topic, k):
    n_row, n_col = np_data.shape
    dkm = np.zeros([n_row, k])
    kwm = np.zeros([n_col, k])
    for n in range(n_row):
        topic_count = []
        for i in range(k):
            row = np_data[n, :]
            counted = np.sum(row[row==i])
            topic_count.append(counted)
        dkm[n, :] = np.array(topic_count)
    for n in range(n_col):
        topic_count = []
        for i in range(k):
            row = np_data[:, n]
            counted = np.sum(row[row==i])
            topic_count.append(counted)
        dkm[n, :] = np.array(topic_count)
    return dkm, kwm

if __name__=="__main__":
    np_data = np.genfromtxt('out_file/test_set.csv', delimiter=',')
    lda(np_data, k)