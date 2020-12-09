# -*- coding = utf-8 -*-
# @Author:何欣泽
# @Time:2020/10/28 12:59
# @File:GUI.py
# @Software:PyCharm

from speech_enhancement import *
import wx
from speech_separation_RNN import get_audio_separation_LSTM
from speech_separation_DNN import *
from add_noise import *


class Frame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title='音频增强', size=(1400, 700), name='frame', style=541072960)
        self.Windows = wx.Panel(self)
        self.Centre()
        self.button()
        self.menu()
        self.spinctrl()

    def spinctrl(self):
        self.adjust = wx.SpinCtrl(self.Windows,size=(60, 20),pos=(820, 585),name='wxSpinCtrl',min=1,max=50,initial=0,style=0)
        self.adjust.SetBase(10)


    def menu(self):
        self.CreateStatusBar()  # A Statusbar in the bottom of the window
        filemenu = wx.Menu()

        menuabout = filemenu.Append(wx.ID_ABOUT, "关于", " Information about this program")
        filemenu.AppendSeparator()
        exit = filemenu.Append(wx.ID_EXIT, "退出", " Terminate the program")
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "帮助")
        self.Bind(wx.EVT_MENU, self.onabout, menuabout)
        self.Bind(wx.EVT_MENU, self.onexit, exit)
        self.SetMenuBar(menuBar)
        self.Show(True)

    '''
    图片设置
    '''
    def picture_reduce(self,event):
        pic1 = wx.Image('./picture/time_pic.png').ConvertToBitmap()
        pic2 = wx.Image('./picture/spectrogram.png').ConvertToBitmap()
        self.pic1 = wx.StaticBitmap(self.Windows, bitmap = pic1,size=(640, 480), pos=(20, 20), name='staticBitmap', style=33554432)
        self.pic2 = wx.StaticBitmap(self.Windows, bitmap = pic2,size=(640, 480), pos=(700, 20), name='staticBitmap', style=33554432)

    def picture_sparation(self,event):
        pic1 = wx.Image('./picture/time_pic.png').ConvertToBitmap()
        pic2 = wx.Image('./picture/spectrogram.png').ConvertToBitmap()
        self.pic1 = wx.StaticBitmap(self.Windows, bitmap = pic1,size=(640, 480), pos=(20, 20), name='staticBitmap', style=33554432)
        self.pic2 = wx.StaticBitmap(self.Windows, bitmap = pic2,size=(640, 480), pos=(700, 20), name='staticBitmap', style=33554432)

    '''
    按钮设置
    '''
    def button(self):
        self.button1 = wx.Button(self.Windows, size=(100, 60), pos=(80, 520), label='选择音频文件', name='button')
        self.button1.Bind(wx.EVT_BUTTON, self.get_path)

        self.button9 = wx.Button(self.Windows, size=(100, 60), pos=(80, 600), label='语音加噪', name='button')
        self.button9.Bind(wx.EVT_BUTTON, self.add_noise)

        self.button2 = wx.Button(self.Windows, size=(100, 60), pos=(200, 520), label='语音降噪', name='button')
        self.button2.Bind(wx.EVT_BUTTON, self.voice_reduce)

        self.button3 = wx.Button(self.Windows, size=(100, 60), pos=(320, 520), label='播放原音频', name='button')
        self.button3.Bind(wx.EVT_BUTTON, self.play_before_audio)

        self.button4 = wx.Button(self.Windows, size=(100, 60), pos=(440, 520), label='播放降噪后音频', name='button')
        self.button4.Bind(wx.EVT_BUTTON, self.play_late_audio)

        self.button5 = wx.Button(self.Windows, size=(100, 60), pos=(560, 520), label='显示图谱', name='button')
        self.button5.Bind(wx.EVT_BUTTON, self.picture_reduce)

        self.button6 = wx.Button(self.Windows, size=(100, 60), pos=(800, 520), label='语音分离DNN', name='button')
        self.button6.Bind(wx.EVT_BUTTON, self.voice_sparation_DNN)

        self.button6 = wx.Button(self.Windows, size=(100, 60), pos=(920, 520), label='语音分离LSTM', name='button')
        self.button6.Bind(wx.EVT_BUTTON, self.voice_sparation_LSTM)

        self.button3 = wx.Button(self.Windows, size=(100, 60), pos=(1040, 520), label='播放原音频', name='button')
        self.button3.Bind(wx.EVT_BUTTON, self.play_before_audio)

        self.button8 = wx.Button(self.Windows, size=(100, 60), pos=(1160, 520), label='播放分离音频', name='button')
        self.button8.Bind(wx.EVT_BUTTON, self.play_sparation_audio)


    def add_noise(self,event):
        output_path = r'C:\Users\MACHENIKE\Desktop\数字信号处理B\项目\noise_set\added_noise.wav'
        add_noise(pathname,output_path)
        dlg = wx.MessageDialog(self, "加噪完成")
        dlg.ShowModal()
        dlg.Destroy()

    '''
    获得音频文件路径
    '''
    def get_path(self, event):
        global pathname

        with wx.FileDialog(self, "Open file",style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            fileDialog.ShowModal()
            pathname = fileDialog.GetPath()

    '''
    开始降噪
    '''
    def voice_reduce(self, event):
        get_audio(path=pathname)
        dlg = wx.MessageDialog(self, "降噪完成")
        dlg.ShowModal()
        dlg.Destroy()

    '''
    语音分离
    '''
    def voice_sparation_DNN(self, event):
        count = self.adjust.GetValue()
        path = pathname
        data, frame = [],[]
        spectrum, en_spectrum = [],[]
        for i in range(count):
            temp,temp_spectrum,temp_en_spectrum,temp_data,temp_frame = get_audio_sparation_DNN(path)
            path = temp
            spectrum.append(temp_spectrum)
            en_spectrum.append(temp_en_spectrum)
            data.append(temp_data)
            frame.append(temp_frame)

        time_pic(data=data[0],s = frame[-1])
        spectrogram(spectrum_early=spectrum[0],spectrum_late=en_spectrum[-1])
        dlg = wx.MessageDialog(self, "DNN分离完成")
        dlg.ShowModal()
        dlg.Destroy()

    def voice_sparation_LSTM(self, event):
        count = self.adjust.GetValue()
        path = pathname
        data, frame = [],[]
        spectrum, en_spectrum = [],[]
        for i in range(count):
            temp,temp_spectrum,temp_en_spectrum,temp_data,temp_frame = get_audio_separation_LSTM(path)
            path = temp
            spectrum.append(temp_spectrum)
            en_spectrum.append(temp_en_spectrum)
            data.append(temp_data)
            frame.append(temp_frame)

        time_pic(data=data[0],s = frame[-1])
        spectrogram(spectrum_early=spectrum[0],spectrum_late=en_spectrum[-1])
        dlg = wx.MessageDialog(self, "LSTM分离完成")
        dlg.ShowModal()
        dlg.Destroy()



    def onabout(self, event):
        dlg = wx.MessageDialog(self, "作者：何欣泽"
                                     "\n邱财旺\n王德天\n王思雅\n韦志成", "音频降噪分离", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def onexit(self, event):
        self.Close(True)

    def play_before_audio(self,event):
        playaudio(path=pathname)

    def play_late_audio(self,event):
        playaudio(path='./output/output_enhancement.wav')

    def play_sparation_audio(self,event):
        playaudio(path='./output/seprartion/seprartion.wav')


class myApp(wx.App):
    def OnInit(self):
        self.frame = Frame()
        self.frame.Show(True)
        return True


if __name__ == '__main__':
    app = myApp()
    app.MainLoop()
