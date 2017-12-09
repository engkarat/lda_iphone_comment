import numpy as np


def update_lda(freq, np_topic, dkm, kwm, row, col, k):
    prop_list = []
    old_topic = np_topic[row, col]
    dkm[row, old_topic] -= freq
    kwm[col, old_topic] -= freq
    for i in range(k):
        # print(dkm[row, i]/np.sum(dkm[row, :]))
        if np.sum(dkm[row, :]) and np.sum(kwm[col, :]):
            prop = (dkm[row, i]/np.sum(dkm[row, :]))*(kwm[col, i]/np.sum(kwm[col, :]))
        else:
            prop = 0
        prop_list.append(prop)
    new_topic = np.argmax(prop_list)
    np_topic[row, col] = new_topic
    dkm[row, new_topic] += freq
    kwm[col, new_topic] += freq

def lda(np_data, k):
    n_row, n_col = np_data.shape
    np_topic = np.random.rand(n_row, n_col)*k
    np_topic = np_topic.astype(np.int)
    dkm, kwm = construct_matrix(np_data, np_topic, k)
    for row in range(n_row):
        for col in range(n_col):
            if np_data[row, col]:
                freq = np_data[row, col]
                update_lda(freq, np_topic, dkm, kwm, row, col, k) 

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
    # hyperparams
    k = 5
    np_data = np.genfromtxt('out_file/test_set.csv', delimiter=',')
    for i in range(500):
        print('Running iteration {}'.format(i+1))
        lda(np_data, k)
    np.savetxt('out_file/dkm.csv', dkm)
    np.savetxt('out_file/kwm.csv', kwm)
