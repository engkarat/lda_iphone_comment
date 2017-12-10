import numpy as np
import logging
import sys
from multiprocessing import Pool
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


def update_lda(freq, np_topic, dkm, kwm, row, col, k):
    prop_list = []
    old_topic = np_topic[row, col]
    dkm[row, old_topic] -= freq
    kwm[col, old_topic] -= freq
    for i in range(k):
        if np.sum(dkm[row, :]) and np.sum(kwm[col, :]):
            prop = (dkm[row, i]/np.sum(dkm[row, :]))*(kwm[col, i]/np.sum(kwm[col, :]))
        else:
            prop = 0
        prop_list.append(prop)
    new_topic = np.argmax(prop_list)
    np_topic[row, col] = new_topic
    dkm[row, new_topic] += freq
    kwm[col, new_topic] += freq

def construct_matrix(np_data, np_topic, k):
    n_row, n_col = np_data.shape
    dkm = np.zeros([n_row, k])
    kwm = np.zeros([n_col, k])
    for n in range(n_row):
        topic_count = []
        for i in range(k):
            row = np_data[n, :]
            counted = np.sum(row[np_topic[n, :]==i])
            topic_count.append(counted)
        dkm[n, :] = np.array(topic_count)
    for n in range(n_col):
        topic_count = []
        for i in range(k):
            row = np_data[:, n]
            counted = np.sum(row[np_topic[:, n]==i])
            topic_count.append(counted)
        kwm[n, :] = np.array(topic_count)
    return dkm, kwm

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
    return dkm, kwm, np_topic

def perplexity(np_data, np_topic, dkm, kwm):
    n_row, n_col = np_data.shape
    accu_prop = []
    for row in range(n_row):
        for col in range(n_col):
            if np_data[row, col]:
                freq = np_data[row, col]
                total_prop = []
                for i in range(k):
                    if np.sum(dkm[row, :]) and np.sum(kwm[col, :]) and dkm[row, i] and kwm[col, i]:
                        prop = (
                            dkm[row, i]/np.sum(dkm[row, :]))*(kwm[col, i]/np.sum(kwm[col, :])
                        )
                        total_prop.append(np.log(prop)*freq)
                    else:
                        total_prop.append(0)
                ans_prop = np.max(total_prop)
                accu_prop.append(ans_prop)
    return np.exp(np.sum(accu_prop)/np.sum(np_data)*-1)

def main(k):
    file_name = 'out_file/tiny_set.csv'
    logging.info("Loading input file : {}".format(file_name))
    with open(file_name) as f:
        total = []
        for i, line in enumerate(f):
            lin = line.split(',')
            total.append(lin)
            if i % 1000 == 0:
                logging.info("Loaded: {}".format(i))
    np_data = np.array(total, dtype=np.int)
    logging.info("Loading completed")
    for i in range(100):
        if (i+1)%20 == 0:
            logging.info('Running iteration {}'.format(i+1))
        dkm, kwm, np_topic = lda(np_data, k)
    perp = perplexity(np_data, np_topic, dkm, kwm)
    logging.info("Perplexity k={} : {}".format(k, perp))

if __name__=="__main__":
    # k = [2,3,4,5,6,7,8,9,10]
    # p = Pool(9)
    # perp_list = p.map(main, k)
    # print(perp_list)
    # try:
    k = sys.argv[1]
    k = int(k)
    main(k)
    # except:
    #     print("Please specify K")
