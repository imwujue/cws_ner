from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model
import re
import pandas as pd
import numpy as np
import codecs
import pickle

maxlen = 200
word_size = 128
batch_size = 1024

def create_model(maxlen, chars, word_size, infer=False):
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    # dropout = Dropout(0.6)(blstm)
    # blstm2 = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(dropout)
    output = TimeDistributed(Dense(6, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    if not infer:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def predict(s, model, chars):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),verbose=False)[0][:len(s)]
        r = np.log(r)
        # nodes = [dict(zip(['D', 'R', 'S', 'B', 'T', 'O'], i[:6])) for i in r]
        # print(nodes)
        res = r.argmax(axis=1)
        tags = []
        for elem in res:
            if elem == 0:
                tags.append('Dis')
            elif elem == 1:
                tags.append('Tre')
            elif elem == 2:
                tags.append('Sym')
            elif elem == 3:
                tags.append('Bod')
            elif elem == 4:
                tags.append('Tes')
            else:
                tags.append('O')
        return tags
    else:
        return []


def test(model, chars):
    file_read = 'data/process/test_cws2.txt'
    file_write = 'data/process/test_res_ner2.txt'
    fw = codecs.open(file_write, 'w', 'utf-8')
    with codecs.open(file_read, 'r', 'utf-8') as f:
        lines = f.readlines()
        # count = 0
        for line in lines:
            # line = re.sub('\s', '', line)
            # if count <= 295 :
            #     count += 1
            #     continue
            # count += 1
            not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
            # cuts = re.compile('\s')
            result = []
            j = 0
            for i in not_cuts.finditer(line):
                # print(line[j:i.start()])
                result.extend(predict(line[j:i.start()], model, chars))
                # print(i.end()-i.start())
                for nums in range(0,i.end()-i.start()):
                    result.append('O')
                j = i.end()
            result.extend(predict(line[j:len(line)-2], model, chars))
            for i in range(0,len(line)-2):
                fw.write(line[i]+'/'+result[i])
                # print(line[i]+'/'+result[i],end="")
            fw.write('\r\n')
            # print()
    fw.close()

def process():
    file_read = 'data/process/test_res_ner2.1.txt'
    file_write = 'data/process/test_res_ner_process.txt'
    fw = codecs.open(file_write, 'w', 'utf-8')
    with codecs.open(file_read, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('/O','',line)
            for elem in re.split('(\s)',line):
                if re.findall('Bod',elem):
                    elem = re.sub('/Bod|/Tre|/Dis|/Tes|/Sym','',elem)
                    elem = '['+elem+']'+'bod'
                    fw.write(elem)
                elif re.findall('Dis',elem):
                    elem = re.sub('/Bod|/Tre|/Dis|/Tes|/Sym','',elem)
                    elem = '['+elem+']'+'dis'
                    fw.write(elem)
                elif re.findall('Tre',elem):
                    elem = re.sub('/Bod|/Tre|/Dis|/Tes|/Sym','',elem)
                    elem = '['+elem+']'+'tre'
                elif re.findall('Sym',elem):
                    elem = re.sub('/Bod|/Tre|/Dis|/Tes|/Sym','',elem)
                    elem = '['+elem+']'+'sym'
                    fw.write(elem)
                elif re.findall('Tes',elem):
                    elem = re.sub('/Bod|/Tre|/Dis|/Tes|/Sym','',elem)
                    elem = '['+elem+']'+'tes'
                    fw.write(elem)
                else:
                    fw.write(elem)
    fw.close()

def MergeTags():
    file_read = 'data/process/test_res_ner_process.txt'
    file_write = 'data/process/test_ner2.txt'
    fw = codecs.open(file_write, 'w', 'utf-8')
    with codecs.open(file_read, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            res = []
            source = re.split('(\s)',line)
            for elem in source:
                # print(elem)
                if re.findall('dis',elem):
                    res.extend('d')
                elif re.findall('bod',elem):
                    res.extend('b')
                elif re.findall('tre',elem):
                    res.extend('r')
                elif re.findall('tes',elem):
                    res.extend('t')
                elif re.findall('sym',elem):
                    res.extend('s')
                else:
                    res.extend('o')
            for r in range(0,len(res)-1):
                # print(res[r],res[r+1])
                # print(source[r],source[r+1])
                if re.findall('bod|tre|dis|tes|sym',source[r]):
                    if res[r] == res[r+2] and res[r+2] == res[r+4]:
                        source[r] = re.sub('bod|tre|dis|tes|sym|\]','',source[r])
                        source[r+2] = re.sub('\[|bod|tre|dis|tes|sym|\]','',source[r+2])
                        source[r+4] = re.sub('\[','',source[r+4])
                        # print(source[r], source[r+2], source[r+4])
                    elif res[r] == res[r+2] and res[r+2] != res[r+4]:
                        source[r] = re.sub('bod|tre|dis|tes|sym|\]', '', source[r])
                        source[r+2] = re.sub('\[', '', source[r+2])
                        # print(source[r],source[r+2])
            for elem in source:
                fw.write(elem)
    fw.close()




if __name__ == '__main__':
    with open('model/chars_ner.pkl', 'rb') as in1:
        chars = pickle.load(in1)
    with open('model/charDF_ner.pkl', 'rb') as in2:
        charDF = pickle.load(in2)
#     file_read = 'ChineseSplit/data/trainset/train_cws.txt'
#     file_write = 'ChineseSplit/data/process/train_cws_tags.txt'
#     print("Start Dataset Processing...")
#     dataset.addTags(file_read, file_write)
#     data, label = dataset.generateLabel(file_write)
#     chars = dataset.generateIndex(data)
#     charDF = dataset.generateDataFrame(chars, data, label)
#     charDF = dataset.paddingDataFrame(charDF)
#     charDF['y'] = charDF['label'].apply(dataset.trans_one)
#     print("Finish Dataset Processing")
    model = create_model(maxlen, chars, word_size)
    print("Start Model Training...")
    # model.load_weights('model/model_ner.h5', by_name=True)
    # history = model.fit(np.array(list(charDF['data_index'])), np.array(list(charDF['one_hot'])).reshape((-1, maxlen, 6)), batch_size=batch_size, epochs=30, verbose=1)
    # model.save('model/model_ner.h5')

    test(model, chars)
    print("Finish Model Training")
    process()
    MergeTags()