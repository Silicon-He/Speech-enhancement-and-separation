# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/10/18 19:14
# @File:speech_enhancement.py
# @Software:PyCharm

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from draw_pic import *


def get_audio_sparation_DNN(path):

    model = load_model('./model/DNNfunction23_model.h5')
    data, fs = librosa.load(path, sr=8000)
    win_length = 256
    hop_length = 64
    nfft = 512

    spectrum = librosa.stft(data, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    magnitude = np.abs(spectrum).T
    phase = np.angle(spectrum).T



    mask = model.predict(magnitude)

    en_magnitude = np.multiply(magnitude, mask)

    en_spectrum = en_magnitude.T * np.exp(1.0j * phase.T)

    frame = librosa.istft(en_spectrum, win_length=win_length, hop_length=hop_length)

    frame = np.multiply(1.5,frame)

    for i in frame:
        if i > 0.6:
            frame[i] = 0.5

    out_file_path = './output/seprartion/seprartion.wav'

    librosa.output.write_wav(out_file_path, frame, sr=8000)

    return out_file_path,spectrum,en_spectrum,data,frame


# if __name__ == '__main__':
#     path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series20.wav'
#     for time in range(5):
#         path = get_audio_sparation(path)
