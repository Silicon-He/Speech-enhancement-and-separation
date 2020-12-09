# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/10/18 19:14
# @File:speech_separation_IRM.py
# @Software:PyCharm


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def get_audio_separation_LSTM(path):

    model = load_model('./model/LSTMfunction23_model(2).h5')
    data, fs = librosa.load(path, sr=8000)
    win_length = 256
    hop_length = 64
    nfft = 512

    spectrum = librosa.stft(data, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    magnitude = np.abs(spectrum).T
    phase = np.angle(spectrum).T



    magnitude_input = np.reshape(magnitude,(1,720,257))
    mask = model.predict(magnitude_input)
    print(np.shape(mask))
    mask = mask[0,:,:]

    print(np.shape(mask))
    en_magnitude = np.multiply(magnitude, mask)
    en_spectrum = en_magnitude.T * np.exp(1.0j * phase.T)


    # spectrogram(spectrum_early=spectrum,spectrum_late=en_spectrum)

    frame = librosa.istft(en_spectrum, win_length=win_length, hop_length=hop_length)

    frame = np.multiply(1.5,frame)

    for i in frame:
        if i > 0.6:
            frame[i] = 0.6
    # time_pic(data, frame)

    out_file_path = './output/seprartion/RNNseprartion.wav'
    librosa.output.write_wav(out_file_path, frame, sr=8000)
    print('输出成功')
    return out_file_path,spectrum,en_spectrum,data,frame


# if __name__ == '__main__':
    # path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series2.wav'
    # for time in range(7):
    #     path = get_audio_separation_LSTM(path)
    # path = get_audio_separation_LSTM(path)
