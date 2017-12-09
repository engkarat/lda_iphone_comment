import json
import pandas as pd

file = open("doc.txt", "r")
doc = json.load(file)
file = open("terms.txt", "r")
terms = json.load(file)
dtm = pd.read_pickle("dtm.pkl")
dzm = pd.read_pickle("dzm.pkl")

### ------------------------   data Loading    ------------------------------------ ###

topic_num = 3
topic_col = []
for i in range(topic_num):
    topic_col.append(str(i))
doc_row = []
for i in range(len(doc)):
    doc_row += str(i)

dkm = pd.DataFrame(index=doc_row, columns=topic_col)   ## initialize DKM
dkm = dkm.fillna(0)

kwm = pd.DataFrame(index=topic_col, columns=terms)     ## initialize KWM
kwm = kwm.fillna(0)
kwm.columns = range(0, len(terms))

### ------------------------   Construct DKM    ------------------------------------ ###

count = 0
for k in range(dkm.columns.size):
    for d in range(dkm.index.size):
        for z in range(dzm.columns.size):
            if dzm.iloc[d, z] == k:
                count += dtm.iloc[d, z]
        dkm.iloc[d, k] = count
        count = 0

### ------------------------   Construct KWM    ------------------------------------ ###

count = 0

for k in range(kwm.index.size):
    for w in range(kwm.columns.size):
        for d in range(dzm.index.size):
            if dzm.iloc[d, w] == k:
                count += dtm.iloc[d, w]
        kwm.iloc[k, w] = count
        count = 0


### ------------------------   From Random to Stable!!    ------------------------------------ ###
prob_list = ["Prob"]
prob_list = pd.DataFrame(index=prob_list, columns=topic_col)
prob_list = prob_list.fillna(0)

alpha = 0.01
beta = topic_num/10

print(dzm, "\n\n", dkm, "\n\n", kwm, "\n\n\n\n\n")

iter_count = 20
for iter in range(5):
    for z in range(dzm.columns.size):
        for d in range(dzm.index.size):
            rm_k = int(dzm.iloc[d, z])                       ## Remove info provided by the Current Word
            dkm.iloc[d, rm_k] -= dtm.iloc[d, z]
            kwm.iloc[rm_k, z] -= dtm.iloc[d, z]

            for kk in range(dkm.columns.size):
                dksum = dkm.iloc[d, :].sum() + (topic_num * alpha)
                dkprob = (dkm.iloc[d, kk] + alpha) / dksum

                kwsum = kwm.iloc[:, z].sum() + (len(terms) * beta)
                kwprob = (kwm.iloc[kk, z] + beta) / kwsum

                prob_kk = dkprob * kwprob
                prob_list.iloc[0, kk] = prob_kk

            s = 0                                             ## find the col with max prob in prob_list
            mxcol = 0
            for i in range(prob_list.columns.size):
                if prob_list.iloc[0, i] > s:
                    s = prob_list.iloc[0, i]
                    mxcol = i

            dzm.iloc[d, z] = mxcol

            add_k = mxcol                                     ## Add back removed values based on NEW assigned K
            dkm.iloc[d, add_k] += dtm.iloc[d, z]
            kwm.iloc[add_k, z] += dtm.iloc[d, z]

    if iter%1 == 0:

        print("finished %i percent of the work!" %iter_count)
        iter_count += 20

print(dzm, "\n\n")
print(dkm, "\n\n")
print(kwm, "\n\n")



