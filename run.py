import dataset
import model as md
import evaluate
import numpy as np
import pandas as pd
import pickle

maxlen = 200
word_size = 128
batch_size = 1024

def Process():
    file_read = 'data/trainset/train_cws.txt'
    file_write = 'data/process/train_cws_tags.txt'
    print("Start Dataset Processing...")
    dataset.addTags(file_read, file_write)
    data, label = dataset.generateLabel(file_write)
    chars = dataset.generateIndex(data)
    charDF = dataset.generateDataFrame(chars, data, label)
    charDF = dataset.paddingDataFrame(charDF)
    charDF['one_hot'] = charDF['label'].apply(dataset.trans_one)
    with open('model/chars.pkl', 'wb') as out1:
        pickle.dump(chars, out1)
    with open('model/charDF.pkl', 'wb') as out2:
        pickle.dump(charDF, out2)
    print("Finish Dataset Processing")
    return chars,charDF

def getData():
    print("Get Processed Dataset...")
    with open('model/chars.pkl', 'rb') as in1:
        chars = pickle.load(in1)
    with open('model/charDF.pkl', 'rb') as in2:
        charDF = pickle.load(in2)
    print("Finish Getting Data")
    return chars,charDF

def train(model):
    print("Start Model Training...")
    history = model.fit(np.array(list(charDF['data_index'])),
                        np.array(list(charDF['one_hot'])).reshape((-1, maxlen, 5)), batch_size=batch_size, epochs=30,
                        verbose=1)
    model.save('model/model.h5')
    print("Finish Model Training")

def test(model, chars):
    print("Start Model Test...")
    model.load_weights('model/model.h5', by_name=True)
    md.test(model, chars)
    print("Finish Model Test")

if __name__ == '__main__':
    #首次运行，处理数据
    # chars,charDF = Process()
    #获取已经处理好的数据
    chars,charDF = getData()
    model = md.create_model(maxlen, chars, word_size)
    # 加载训练好的模型
    # model.load_weights('model/model.h5', by_name=True)
    #重新训练模型
    # train(model)
    test(model, chars)
    # evaluate.evaluate()