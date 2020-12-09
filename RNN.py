# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/11/4 17:31
# @File:RNN.py
# @Software:PyCharm


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
import librosa


def generateDataset(woman_path, mixed_path):
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
    mask = IRM(woman_mag, mix_mag)

    return mix_mag, mask


def IRM(clean_spectrum, mix_spectrum):

    snr = np.divide(np.abs(clean_spectrum), np.abs(mix_spectrum))
    # IRM
    mask = snr / (snr + 1)
    mask[np.isnan(mask)] = 0.5
    mask = np.power(mask, 0.5)

    return mask


def get_model():
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(512, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(keras.layers.LSTM(1024, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))


    model.add(keras.layers.Dense(257))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    return model


def train(model, train_x, train_y, text_x, text_y):

    model.compile(loss='mse', optimizer='adam', metrics=['mse'], )

    cheakpoint_save_path = './cheakpoint/LSTMfunction23(2).ckpt'


    if os.path.exists(cheakpoint_save_path + '.index'):
        print('-------------load the model-----------')
        model.load_weights(cheakpoint_save_path)


    RNN_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cheakpoint_save_path,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      monitor='val_loss')

    history = model.fit(train_x, train_y,
                        batch_size=1, epochs=100, validation_split=0.,
                        validation_data=(text_x, text_y),
                        validation_freq=5,
                        callbacks=[RNN_callback])

    model.save("./model/LSTMfunction23_model(2).h5")
    print(model.summary())

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return loss, val_loss


def main():
    global train_x, train_y, text_x, text_y
    num = 1
    cout = 1
    for i in range(1, 30):
        clean_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech{}.wav'.format(i)
        mix_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series{}.wav'.format(i)
        feature, label = generateDataset(clean_path, mix_path)
        if np.shape(feature[:, 0]) == (720,):
            print(i)
            if cout == 2:
                train_x = [feature, train_x]
            elif cout == 1:
                train_x = feature
            else:
                train_x = np.insert(train_x, 0, feature, axis=0)

        if np.shape(label[:, 0]) == (720,):
            if cout == 2:
                train_y = [label, train_y]
            elif cout == 1:
                train_y = label
            else:
                train_y = np.insert(train_y, 0, label, axis=0)
        cout = cout + 1

    print(np.shape(train_x),np.shape(train_y))


    for j in range(30, 33):
        clean_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech{}.wav'.format(j)
        mix_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series{}.wav'.format(j)
        feature, label = generateDataset(clean_path, mix_path)
        if np.shape(feature[:, 0]) == (720,):
            if num == 2:
                text_x = [feature, text_x]
            elif num == 1:
                text_x = feature
            else:
                text_x = np.insert(text_x, 0, feature, axis=0)

        if np.shape(label[:, 0]) == (720,):
            if num == 2:
                text_y = [label, text_y]
            elif num == 1:
                text_y = label
            else:
                text_y = np.insert(text_y, 0, label, axis=0)
        num = num + 1
    print(np.shape(text_x),np.shape(text_y))


    print('------------------------start training------------------')
    model = get_model()
    loss, val_loss = train(model,train_x, train_y, text_x, text_y)

if __name__ == '__main__':
    main()
