import numpy as np
from scipy import signal
from ._math import standarize


def do_fft(wave_vec):
    N = len(wave_vec)            # サンプル数
    dt = 1          # サンプリング間隔
    t = np.arange(0, N*dt, dt) # 時間軸
    freq = np.linspace(0, 1.0/dt, N) # 周波数軸

    f = wave_vec
    F = np.fft.fft(f)

    # 振幅スペクトルを計算
    Amp = np.abs(F)
    return F

def make_spectrum(wave_vec):
    F = do_fft(wave_vec)
    spectrum = np.abs(F)**2
    spectrum = spectrum[:len(spectrum)//2]
    return standarize(spectrum)



def decode(spe):

    length = len(spe)
    mid = length//2

    real_ = spe[:mid]
    imag_ = spe[mid:]

    c_list = []
    for i in range(len(imag_)):
        c_list.append(complex(0,imag_[i]))

    c = np.array(c_list)
    F = real_ + c
    return F

def norm(spectrum):
    N = len(spectrum)
    spectrum = spectrum / (N/2)
    return spectrum


def butter_lowpass(lowcut, fs, order=4):
    '''
    バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a

def butter_lowpass_filter(x, lowcut, fs, order=4):
    '''データにローパスフィルタをかける関数
    '''
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y

def butter_highpass( highcut, fs, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(self, x, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y