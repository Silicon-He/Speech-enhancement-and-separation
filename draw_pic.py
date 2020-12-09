#-*- coding = utf-8 -*-
#@Author:何欣泽
#@Time:2020/11/6 9:58
#@File:draw_pic.py
#@Software:PyCharm

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

#时域图
def time_pic(data, s):
    plt.figure(2)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    plt.sca(ax1)
    plt.plot(data)
    plt.sca(ax2)
    plt.plot(s)
    plt.savefig('./picture/time_pic.png')
    plt.close()



#画语谱图
def spectrogram(spectrum_early,spectrum_late):
    #初始化画布
    plt.figure(2)

    #去噪前语谱图
    D = librosa.amplitude_to_db(spectrum_early, ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('spectrogram_before')

    #去噪后语谱图
    D = librosa.amplitude_to_db(spectrum_late, ref=np.max)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('spectrogram_later')

    #显示
    plt.savefig('./picture/spectrogram.png')
    plt.close()
