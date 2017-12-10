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

test_set = reloaded[10000:11000, :]
np.savetxt('out_file/sampled_small_set.csv', test_set, fmt='%d', delimiter=',')
