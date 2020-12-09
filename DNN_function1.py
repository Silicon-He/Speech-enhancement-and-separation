# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/10/18 15:17
# @File:DNN.py
# @Software:PyCharm

import numpy as np
import librosa
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

def generateDataset(mix_path,clean_path):
    mix, sr = librosa.load(mix_path, sr=8000)
    clean, sr = librosa.load(clean_path, sr=8000)

    win_length = 256
    hop_length = 128
    nfft = 512

    mix_spectrum = librosa.stft(mix, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    clean_spectrum = librosa.stft(clean, win_length=win_length, hop_length=hop_length, n_fft=nfft)


    mix_mag = np.abs(mix_spectrum).T
    clean_mag = np.abs(clean_spectrum).T


    frame_num = mix_mag.shape[0] - 4

    feature = np.zeros([frame_num, 257 * 5])
    k = 0
    for i in range(frame_num - 4):
        frame = mix_mag[k:k + 5]
        feature[i] = np.reshape(frame, 257 * 5)
        k += 1

    snr = np.divide(clean_mag, mix_mag)
    mask = np.around(snr,0)
    mask[np.isnan(mask)] = 1
    mask[mask > 1] = 1

    label = mask[2:-2]

    ss = StandardScaler()
    feature = ss.fit_transform(feature)
    return feature, label


def getModel():

    model = Sequential()

    model.add(Dense(2048, input_dim=1285))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))


    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))


    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))


    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))


    model.add(Dense(257))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model


def train(feature, label, text_x, text_y, model):

    model.compile(optimizer='adam',loss='mse',metrics=['mse'])

    cheakpoint_save_path = './cheakpoint/DNNfunction1.ckpt'


    if os.path.exists(cheakpoint_save_path + '.index'):
        print('-------------load the model-----------')
        model.load_weights(cheakpoint_save_path)


    DNN_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cheakpoint_save_path,
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_loss')

    history = model.fit(feature, label,
                        batch_size=128, epochs=5, validation_split=0.,
                        validation_data=(text_x,text_y),
                        validation_freq=5,
                        callbacks=[DNN_callback])

    model.save("./model/DNNfunction1_model.h5")


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return loss,val_loss


def plot(loss,val_loss):

    plt.plot(loss,label = 'Training loss')
    plt.scatter(range[0:20:2],val_loss,label = 'Texting loss')
    plt.title('Training and Texting loss')
    plt.legend()
    plt.show()



def main():
    global train_x,train_y,text_x,text_y

    for i in range(1,10):
        cout = i
        clean_path = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\test_data_set\data (" + str(cout) + ").wav"
        mix_path = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\noise_" + str(cout) + '.wav'
        feature, label = generateDataset(mix_path,clean_path)
        if cout == 1:
            train_x = feature
            train_y = label
        else:
            train_x = np.vstack((train_x,feature))
            train_y = np.vstack((train_y,label))

    print('----------------train data succeed-----------------')
    for n in range(81,85):
        cout = n
        clean_path = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\test_data_set\data (" + str(cout) + ").wav"
        mix_path = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\noise_" + str(cout) + '.wav'
        feature, label = generateDataset(mix_path,clean_path)
        if cout == 81:
            text_x = feature
            text_y = label
        else:
            text_x = np.vstack((text_x,feature))
            text_y = np.vstack((text_y,label))


    print('------------------------start training------------------')
    model = getModel()
    loss,val_loss = train(train_x, train_y, text_x,text_y, model)
    print('-----------------------plot---------------------------')
    print(loss,val_loss)


if __name__ == "__main__":
    main()
