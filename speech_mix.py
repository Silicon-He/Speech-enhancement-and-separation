#-*- coding = utf-8 -*-
#@Author:何欣泽
#@Time:2020/11/3 20:09
#@File:speech_mix.py
#@Software:PyCharm

import numpy as np
import librosa

for cout in range(1,34):
    woman = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_voice\woman_voice ({cout}).wav".format(cout = cout)
    man = r"C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\man_voice\man_voice (1).wav".format(cout = cout)

    woman_speech ,sample_rate1,= librosa.load(woman, sr=8000)
    man_speech ,sample_rate2,= librosa.load(man, sr=8000)

    # 找最短的音频
    minlength = min(len(woman_speech), len(man_speech))

    # 裁剪
    woman_speech = woman_speech[:minlength]
    man_speech = man_speech[:minlength]
    # 线性相加
    mixed_series = woman_speech + man_speech

    mixed_file_name = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\mixed_series\mixed_series{}.wav'.format(cout)
    woman_file_name = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\woman_series\woman_speech{}.wav'.format(cout)
    man_file_name = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\man_series\man_speech{}.wav'.format(cout)
    #保存文件
    librosa.output.write_wav(mixed_file_name, mixed_series, sr=8000)
    librosa.output.write_wav(woman_file_name, woman_speech, sr=8000)
    librosa.output.write_wav(man_file_name, man_speech, sr=8000)

print('合成完成')