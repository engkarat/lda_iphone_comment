import numpy as np
with open('out_file/cleaned_data_matrix.csv') as f:
    total = []
    for i, line in enumerate(f):
        lin = line.split(',')
        total.append(lin)
        if i % 10000 == 0:
            print(i)

reloaded = np.array(total, dtype=np.int)
print(reloaded.shape)

test_set = reloaded[25000:25500, 25000:25500]
np.savetxt('out_file/test_set.csv', test_set, fmt='%d', delimiter=',')