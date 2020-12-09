# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/10/18 19:14
# @File:speech_enhancement.py
# @Software:PyCharm

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import winsound
from draw_pic import *

def get_audio(path):

    model = load_model('./model/DNNfunction1_model.h5')
    data, fs = librosa.load(path, sr=8000)
    win_length = 256
    hop_length = 128
    nfft = 512

    spectrum = librosa.stft(data, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    magnitude = np.abs(spectrum).T
    phase = np.angle(spectrum).T


    frame_num = magnitude.shape[0] - 4
    feature = np.zeros([frame_num, 257 * 5])
    k = 0
    for i in range(frame_num - 4):
        frame = magnitude[k:k + 5]
        feature[i] = np.reshape(frame, 257 * 5)
        k += 1

    #二值掩码
    ss = StandardScaler()
    feature = ss.fit_transform(feature)
    mask = model.predict(feature)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    magnitude = magnitude[2:-2]
    en_magnitude = np.multiply(magnitude, mask)
    phase = phase[2:-2]

    en_spectrum = en_magnitude.T * np.exp(1.0j * phase.T)

    spectrogram(spectrum_early=spectrum,spectrum_late=en_spectrum)

    frame = librosa.istft(en_spectrum, win_length=win_length, hop_length=hop_length)

    time_pic(data, frame)
    librosa.output.write_wav("./output/output_enhancement.wav", frame, sr=8000)


def playaudio(path):
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

if __name__ == '__main__':
    get_audio(r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\noise_2.wav')