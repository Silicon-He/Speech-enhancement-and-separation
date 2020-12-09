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


def generateDataset(woman_path, mixed_path):
    # woman_file = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech1.wav'
    # mixed_file = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series1.wav'
    # man_file = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\man_series\man_speech1.wav'

    samples_woman, _ = librosa.load(woman_path, sr=8000)
    # samples_man, _ = librosa.load(man_file, sr=8000)
    mixed_series, _ = librosa.load(mixed_path, sr=8000)

    win_length = 256
    hop_length = 64
    nfft = 512

    mix_spectrum = librosa.stft(mixed_series, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    woman_spectrum = librosa.stft(samples_woman, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    # man_spectrum = librosa.stft(samples_man, win_length=win_length, hop_length=hop_length, n_fft=nfft)

    woman_mag = np.abs(woman_spectrum.T)
    mix_mag = np.abs(mix_spectrum.T)
    mask = IRM(woman_mag,mix_mag)

    return mix_mag,mask

def IRM(clean_spectrum, mix_spectrum):

    snr = np.divide(np.abs(clean_spectrum), np.abs(mix_spectrum))
    # IRM
    mask = snr / (snr + 1)
    mask[np.isnan(mask)] = 0.5
    mask = np.power(mask, 0.5)

    return mask


def getModel():

    model = Sequential()

    model.add(Dense(2048, input_dim=257))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))


    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(257))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model


def train(feature, label, text_x, text_y, model):
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    cheakpoint_save_path = './cheakpoint/DNNfunction23(1).ckpt'


    if os.path.exists(cheakpoint_save_path + '.index'):
        print('-------------load the model-----------')
        model.load_weights(cheakpoint_save_path)

    DNN_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cheakpoint_save_path,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      monitor='val_loss')

    history = model.fit(feature, label,
                        batch_size=128, epochs=80, validation_split=0.,
                        validation_data=(text_x, text_y),
                        validation_freq=5,
                        callbacks=[DNN_callback])

    model.save("./model/DNNfunction23_model.h5")


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return loss, val_loss



def main():
    global train_x, train_y, text_x, text_y

    for i in range(1, 30):
        cout = i
        clean_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech{}.wav'.format(cout)
        mix_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series{}.wav'.format(cout)
        feature, label = generateDataset(clean_path,mix_path)
        if cout == 1:
            train_x = feature
            train_y = label
        else:
            train_x = np.vstack((train_x, feature))
            train_y = np.vstack((train_y, label))
            print(np.shape(train_x),np.shape(train_y))

    for i in range(30, 33):
        cout = i
        clean_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech{}.wav'.format(cout)
        mix_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series{}.wav'.format(cout)
        feature, label = generateDataset(clean_path,mix_path)
        if cout == 30:
            text_x = feature
            text_y = label
        else:
            text_x = np.vstack((text_x, feature))
            text_y = np.vstack((text_y, label))
            print(np.shape(text_x),np.shape(text_y))


    print('------------------------start training------------------')
    model = getModel()
    loss, val_loss = train(train_x, train_y, text_x, text_y, model)



if __name__ == "__main__":
    main()
