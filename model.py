from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model, Sequential
import re
import pandas as pd
import numpy as np
import codecs
from keras_contrib.layers.crf import CRF

maxlen = 200
word_size = 128
batch_size = 1024

def create_model(maxlen, chars, word_size, infer=False):
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    dropout = Dropout(0.6)(blstm)
    blstm2 = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(dropout)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm2)

    model = Model(input=sequence, output=output)
    if not infer:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model = Sequential()
    # model.add(Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True))
    # model.add(Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(32)))
    # crf = CRF(5,sparse_target=True)
    # model.add(crf)
    # model.summary()
    # model.compile('adam',loss = crf.loss_function, metrics=[crf.accuracy])
    return model

def viterbi(nodes):
    zy = {'be': 0.5,
          'bm': 0.5,
          'eb': 0.5,
          'es': 0.5,
          'me': 0.5,
          'mm': 0.5,
          'sb': 0.5,
          'ss': 0.5
          }
    zy = {i: np.log(zy[i]) for i in zy.keys()}
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[
                        path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]  # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def simpleCut(s, model, chars):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                          verbose=False)[
                0][:len(s)]
        r = np.log(r)
        # if r==0:
        #     r = -9999
        # else:
        #     r = np.log(r)
        # print(r)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        # print(nodes)
        t = viterbi(nodes)
        # print("t",t)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        # print("words", words)
        return words
    else:
        return []


def wordCut(s, model, chars):
    result = []
    j = 0
    not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
    for i in not_cuts.finditer(s):
        result.extend(simpleCut(s[j:i.start()], model, chars))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simpleCut(s[j:], model, chars))
    return result

def test(model, chars):
    file_read = 'data/testset1/test2.txt'
    file_write = 'data/process/test_cws2.1.txt'
    fw = codecs.open(file_write, 'w', 'utf-8')
    count =0
    with codecs.open(file_read, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            count += 1
            if count!=329:
                continue
            line = re.sub('\s', '', line)
            # print(line)
            # print(wordCut(line, model, chars))
            wordList = wordCut(line, model, chars)
            for word in wordList:
                # print(word + ' ')
                fw.write(word + ' ')
            fw.write('\r\n')
    fw.close()
