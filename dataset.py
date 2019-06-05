import codecs
import re
import numpy as np
import pandas as pd
from keras.utils import np_utils

maxlen = 200

def addTags(file_read, file_write):
    f = codecs.open(file_write,'w','utf-8')
    # count = 0
    with codecs.open(file_read, 'r', 'utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            # count += 1
            word_list = line.strip().split()
            for word in word_list:
                if len(word) == 1:
                    f.write(word+'/S'+' ')
                else:
                    f.write(word[0]+'/B'+' ')
                    for w in word[1:-1]:
                        f.write(w+'/M'+' ')
                    f.write(word[-1]+'/E'+' ')
            f.write('\r\n')
            # if count == 10:
            #     break
    f.close()

def getLabel(sen):
    sen = re.findall('(.)/(.)',sen)
    if sen:
        sen = np.array(sen)
        return list(sen[:,0]), list(sen[:,1])

def generateLabel(file_write):
    data = []
    label = []
    with codecs.open(file_write, 'r', 'utf-8') as fr:
        # sentences = fr.read().split('\r\n')
        sentences = re.split(u'[，。！？、]', fr.read())
        # sentences = fr.read().split('。')
        for sen in sentences:
            res = getLabel(sen)
            if res:
                data.append(res[0])
                label.append(res[1])
    return data,label

def generateIndex(data):
    chars = []
    for elem in data:
        chars.extend(elem)
    chars = pd.Series(chars).value_counts()
    # print(chars)
    chars[:]=range(1, len(chars)+1)
    # print(chars)
    return chars

def generateDataFrame(chars, data, label):
    charDF = pd.DataFrame(index=range(len(data)))
    charDF['data'] = data
    charDF['label'] = label
    # print(charDF['label'][0])
    # charDF['label'].apply(lambda x: x.extend('X'*(maxlen - len(x))))
    # print(charDF['label'][0])
    charDF.index = range(len(charDF))
    charDF = charDF[charDF['data'].apply(len) <= maxlen]
    charDF = charDF[charDF['label'].apply(len) <= maxlen]
    # print(charDF['data'])
    # print(charDF['label'])
    charDF['data_index']=charDF['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))
    # print(charDF['data_index'])
    return charDF

def paddingDataFrame(charDF):
    # tag = pd.Series({'S': 0, 'B': 1, 'M': 2, 'E': 3, 'X':4})
    charDF = charDF[charDF['label'].apply(len) <= maxlen]
    # print(charDF['label'])
    # charDF['one_hot'] = charDF['label'].map(lambda x: tag[x].values.reshape(-1,1))
    # print(charDF['one_hot'])
    # for i in range(0,len(charDF)):
    #     for j in range(0,len(charDF['one_hot'][i])):
    #         if (charDF['one_hot'][i][j] != 0 and charDF['one_hot'][i][j] != 1 and charDF['one_hot'][i][j] != 2 and charDF['one_hot'][i][j] != 3):
    #             charDF['one_hot'][i][j] = 4
    #     for j in range(len(charDF['one_hot'][i]),maxlen):
    #         np.append(charDF['one_hot'][i],4)
    for i in range(0,len(charDF)):
        for j in range(0,len(charDF['label'][i])):
            # print(len(charDF['label'][i]))
            if (charDF['label'][i][j] != 'S' and charDF['label'][i][j] != 'B' and charDF['label'][i][j] != 'M' and charDF['label'][i][j] != 'E'):
                charDF['label'][i][j] = 'X'
        for j in range(len(charDF['label'][i]),maxlen):
            np.append(charDF['label'][i],'X')
    #     charDF['one_hot'][i] = list(np_utils.to_categorical(charDF['one_hot'][i], 5))
    # print(charDF['label'])
    return charDF

def trans_one(x):
    tag = pd.Series({'S': 0, 'B': 1, 'M': 2, 'E': 3, 'X': 4})
    _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)

