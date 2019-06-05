import codecs
import numpy as np
import pandas as pd
import re
import pickle
from keras.utils import np_utils

maxlen = 200

def addTags(file_read, file_write):
    f = codecs.open(file_write, 'w', 'utf-8')
    # count =0
    with codecs.open(file_read,'r','utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = re.sub('\s', '', line)
            print(line)
            flag = 0
            sentences = re.split('(\\['+'.*?'+'\\]dis|\\['+'.*?'+'\\]tre|\\['+'.*?'+'\\]sym|\\['+'.*?'+'\\]tes|\\['+'.*?'+'\\]bod)',line)
            for sen in sentences:
                print(sen)
                if sen and sen[0]!='[':
                    for char in sen:
                        f.write(char + '/O' + ' ')
                elif re.findall('dis',sen):
                    sen = re.sub('\\[|\\]|dis|bod','',sen)
                    for char in sen:
                        f.write(char + '/D' + ' ')
                elif re.findall('tre',sen):
                    sen = re.sub('\\[|\\]|tre|bod','',sen)
                    for char in sen:
                        f.write(char + '/R' + ' ')
                elif re.findall('sym',sen):
                    sen = re.sub('\\[|\\]|sym|bod','',sen)
                    for char in sen:
                        f.write(char + '/S' + ' ')
                elif re.findall('tes',sen):
                    sen = re.sub('\\[|\\]|tes|bod','',sen)
                    for char in sen:
                        f.write(char + '/T' + ' ')
                elif re.findall('bod',sen):
                    sen = re.sub('\\[|\\]|bod','',sen)
                    for char in sen:
                        f.write(char + '/B' + ' ')
            f.write('\r\n')
            # count += 1
            # if count == 3:
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
        sentences = re.split(u'[，。！？、]', fr.read())
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
    chars[:]=range(1, len(chars)+1)
    return chars

def generateDataFrame(chars, data, label):
    charDF = pd.DataFrame(index=range(len(data)))
    charDF['data'] = data
    charDF['label'] = label
    charDF.index = range(len(charDF))
    charDF = charDF[charDF['data'].apply(len) <= maxlen]
    charDF = charDF[charDF['label'].apply(len) <= maxlen]
    charDF['data_index']=charDF['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))
    return charDF

def paddingDataFrame(charDF):
    charDF = charDF[charDF['label'].apply(len) <= maxlen]
    for i in range(0,len(charDF)):
        for j in range(0,len(charDF['label'][i])):
            # print(len(charDF['label'][i]))
            if (charDF['label'][i][j] != 'D' and charDF['label'][i][j] != 'R' and charDF['label'][i][j] != 'S' and charDF['label'][i][j] != 'B' and charDF['label'][i][j] != 'T'):
                charDF['label'][i][j] = 'O'
        for j in range(len(charDF['label'][i]),maxlen):
            np.append(charDF['label'][i],'O')
    return charDF

def trans_one(x):
    tag = pd.Series({'D':0, 'R':1, 'S':2, 'B':3, 'T':4, 'O':5})
    _ = map(lambda y: np_utils.to_categorical(y, 6), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)

if __name__ == '__main__':
    file_read = 'data/trainset/train_ner.txt'
    file_write = 'data/process/train_ner_tags.txt'
    addTags(file_read, file_write)
    data, label = generateLabel(file_write)
    chars = generateIndex(data)
    charDF = generateDataFrame(chars, data, label)
    charDF = paddingDataFrame(charDF)
    charDF['one_hot'] = charDF['label'].apply(trans_one)
    with open('model/chars_ner.pkl','wb') as out1:
        pickle.dump(chars,out1)
    with open('model/charDF_ner.pkl','wb') as out2:
        pickle.dump(charDF,out2)
    print(charDF)