
# coding: utf-8

# In[2]:

import json
import nltk
import sklearn
import string
import numpy as np
from nltk.corpus import stopwords


# In[3]:

with open('out_file/contents_data.json') as f:
    contents_data_json = f.read()
contents_data = json.loads(contents_data_json)


# In[4]:

print(len(contents_data))
print(contents_data[195])


# In[5]:

eng_stopwords = set(stopwords.words('english'))


# In[50]:

def clean_txt(txt):
    txt = txt.lower()
    for i in string.punctuation:
        txt = txt.replace(i, ' ')
    txt_token = txt.split(' ')
    res_token = []
    for txt in txt_token:
        if txt and txt not in eng_stopwords:
            res_token.append(txt)
    res = " ".join(res_token)
    return res


# In[51]:

clean_txt("earth!, test i you & I will test it later")


# In[52]:

contents_cleaned = []
for content_data in contents_data:
    content_cleaned = {}
    content_cleaned['comment'] = clean_txt(content_data['comment'])
    content_cleaned['time'] = content_data['time']
    contents_cleaned.append(content_cleaned)
print(len(contents_cleaned))


# In[53]:

contents_cleaned[0]


# In[54]:

cnt_vec = sklearn.feature_extraction.text.CountVectorizer()
text = []
for content_cleaned in contents_cleaned:
    text.append(content_cleaned['comment'])
text = text[10000:15000]
cnt_vec.fit(text)


# In[55]:

res = cnt_vec.transform(text)
print(res.shape)


# In[56]:

len(cnt_vec.vocabulary_)


# In[57]:

vocabulary_inverse = dict((v,k) for k,v in cnt_vec.vocabulary_.iteritems())
len(vocabulary_inverse)


# In[58]:

n = 4
tmp = res[n].toarray()
for i in range(tmp.shape[1]):
    if tmp[0, i]:
        print vocabulary_inverse[i], tmp[0, i]
print(contents_cleaned[n])
print(contents_data[n])


# In[59]:

np_res = res.toarray()


# In[69]:

np_res[0, 0]


# In[71]:

np.savetxt('out_file/small_5000.csv', np_res, fmt='%d', delimiter=',')


# In[72]:

# vocab = []
# for i in range(len(vocabulary_inverse)):
#     vocab.append(vocabulary_inverse[i])


# In[78]:

# with open('out_file/vocab.txt', 'w+') as f:
#     for i in vocab:
#         f.write(i.encode('utf8')+'\n')


# In[83]:

# with open('out_file/cleaned_data_matrix.csv') as f:
#     total = []
#     for i, line in enumerate(f):
#         lin = line.split(',')
#         total.append(lin)
#         if i % 10000 == 0:
#             print(i)
# reloaded = np.array(total, dtype=np.int)
# reloaded.shape


# In[ ]:



