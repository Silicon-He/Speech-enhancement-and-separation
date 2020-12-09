# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/11/4 22:22
# @File:test.py
# @Software:PyCharm


import numpy as np
import librosa

def add_noise(input_path,output_path):

    data, fs = librosa.core.load(input_path)

    wn = np.random.normal(0, 1, len(data))

    data_noise = np.where(data != 0.0, data + 0.02 * wn, 0.0).astype(np.float32) #加噪声


    librosa.output.write_wav(output_path, data_noise, fs)

# path_out = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\turely_noise.wav'
# path_input = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\text.wav"
# add_noise(path_input,path_out)