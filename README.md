# Speech-enhancement-and-separation
大三秋季学习DSP课程项目，语音增强和分离。

原理是基于mask的语音信号增强分离，用神经网络训练完成。

语音增强是用深度神经网络DNN，mask用的是理想二值掩膜IBM；

语音分离用DNN和长短时记忆神经网络LSTM两个模型对比完成，mask用的是理想比值掩蔽IRM；

其中经测试DNN训练快，泛化能力差，LSTM相反。

其中运用扩帧的操作和网络构建有参考，参考代码为：https://github.com/Ryuk17/SpeechAlgorithms
