## ----------- To Reduce the noise from the recorded audio ----------------------

## STILL NEEDS FORMATING AND MAKE IT MORE UNDERSTANDABLE

'''
Aim: Applying an appropriate filter on the signal to reduce the noise 
Number of Test : 05
Records Path : Analysis\Filter_comparing
Filters : 
    - Butter Worth Band Pass Filters (on raw captured audio)            <----  SNR(avg): -97.64   |   RMSE(avg):  3,395.15  ---->  
    - EMD ICEEMDAN Filters  (on raw captured audio)                     <----  SNR(avg): -84.06   |   RMSE(avg):  1,715.90  ---->
    - EMD ICEEMDAN Filters ( on Butterworth bp filter)                  <----  SNR(avg): -91.27   |   RMSE(avg):  2,486.32  ---->
Observed: 
    - EMD ICEEMDAN Filters works well as compared to other

'''

# Import the Dependencies
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as ss
from PyEMD import CEEMDAN
from PyEMD import EMD
import numpy as np
import pandas as pd
import tkinter as tk
import pywt
import time
 
# -----------------------------   GLOBAL PARAMETER   ---------------------------------------
SAMPLE_RATE = 4000
LOWCUT = 1
HIGHCUT = 1999
ORDER = 5
WINDOW_SIZE = 20
FL_HZ = 10
RIPPLE_DB = 10.0

# ------------------------------   RECORD AUDIO SIGNAL   ------------------------------------
def capture_audio(duration: int):
    print('Recording....')
    audio_data = sd.rec(int(duration * SAMPLE_RATE),SAMPLE_RATE, channels=1, dtype = 'int16' )
    sd.wait()
    print('Processing...')
    return audio_data.flatten(), SAMPLE_RATE

# Function to display countdown window
def show_countdown():
    """
    Displays a countdown window using Tkinter.
    """
    countdown_window = tk.Tk()
    countdown_window.title("Countdown")
    countdown_label = tk.Label(countdown_window, text="", font=("Arial", 24))
    countdown_label.pack()

    for i in range(5, 0, -1):
        countdown_label.config(text=f"Get ready!\nStarting in {i}...")
        countdown_window.update()
        time.sleep(1)

    countdown_window.destroy()

# --------------------------------------------------   BUTTERWORTH BANDPASS (100 - 1000)    ----------------------------------------------

def butter_bp(lowcut: int, highcut: int, sr: int, order = 5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ss.butter(order, [low, high], btype='band')
    return b, a

def butterworth_filter(signal: list, lowcut: int, highcut: int, order: int, sr: int):
    b, a = butter_bp(lowcut, highcut, sr, order)
    y = ss.lfilter(b, a, signal)
    return y

# -------------------------------------------------------------  ICEEMDAN FILTER  --------------------------------------------------------

def iceemdan_filter(signal: list):
    iceemdan = CEEMDAN()
    IMFS = iceemdan(signal)
    denoised_signal = signal - IMFS[0]
    return denoised_signal, IMFS[0]

def emd_filter(signal: list):
    emd = EMD()
    IMFS = emd.emd(signal)
    denoised_signal = signal - IMFS[0]
    return denoised_signal, IMFS[0]

def wavelet_denoising(signal: list) -> list:
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, 'db1')
    return denoised_signal

# ---------------------------------------------------------------- EVALUATION -----------------------------------------------------------

def calculate_snr(signal: np.ndarray, denoise_signal: np.ndarray) -> float:
    noise = signal - denoise_signal
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log(signal_power / noise_power)
    return snr

def calculate_rmse(signal: list, denoise_signal: list) -> float:
    rmse = np.sqrt(np.mean((signal - denoise_signal)** 2))
    return rmse

#  ------------------------------------------------------------ COMPARING FILTERS --------------------------------------------------------

def plot_signal(time: list, raw_audio_signal: list, butter_bp_filter: list, denoised_signal: list ,B_denoised_signal, sr: int, rmse: list, snr: list ):
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(time, raw_audio_signal, label = 'Original Audio Signal', color = 'b')
    plt.plot(time, butter_bp_filter, label = 'Butter Worth Bandpass Audio Signal', color = 'r', alpha = 0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Original Signal V/s ButterWorth Signal')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(butter_bp_filter, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'ButterWorth Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[0]:.4f} | SNR: {snr[0]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2, 3)
    plt.plot(time, raw_audio_signal, label = 'Original Audio Signal', color = 'b')
    plt.plot(time, denoised_signal, label = 'ICEEMDAN Filtered Audio', color = 'r', alpha = 0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Original Signal V/s IMF0(EMD) Signal')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.subplot(3, 2 , 4)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(denoised_signal, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'IMF0 Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[1]:.4f} | SNR: {snr[1]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2, 5)
    plt.plot(time, raw_audio_signal, label = 'Original Audio Signal', color = 'b')
    plt.plot(time, B_denoised_signal, label = 'EMD Filtered Audio ', color = 'r', alpha = 0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude') 
    plt.title('Original Signal V/s Denoised Signal(EMD)')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.subplot(3, 2 , 6)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(B_denoised_signal, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'Denoised Audio Signal(without butterBP)', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[2]:.4f} | SNR: {snr[2]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout(pad=3.0) 
    plt.show()


def plot_psd(raw_audio_signal: list, raw_audio_signal2: list ,butter_bp_filter: list, imf0: list ,B_denoised_signal, sr: int, rmse: list, snr: list,A_rmse: list, A_snr: list ):
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(butter_bp_filter, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'ButterWorth Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {A_rmse[0]:.4f} | SNR: {A_snr[0]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2, 3)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(imf0, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'ICEEMDAN Filtered Audio', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {A_rmse[1]:.4f} | SNR: {A_snr[1]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2, 5)
    f, Pxx = ss.welch(raw_audio_signal, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(B_denoised_signal, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Original Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'EMD Audio Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {A_rmse[2]:.4f} | SNR: {A_snr[2]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2 , 2)
    f, Pxx = ss.welch(raw_audio_signal2, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(butter_bp_filter, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'ButterWorth Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'ButterWoth Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[0]:.4f} | SNR: {snr[0]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2, 4)
    f, Pxx = ss.welch(raw_audio_signal2, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(imf0, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'Butter worth Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'ICEEMDAN Filtered Audio', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[1]:.4f} | SNR: {snr[1]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.subplot(3, 2 , 6)
    f, Pxx = ss.welch(raw_audio_signal2, fs=sr, nperseg=1024)
    fd, Pxxd = ss.welch(B_denoised_signal, fs = sr, nperseg=1024)
    plt.semilogy(f, Pxx, label = 'ButterWorth Signal',alpha = 1)
    plt.semilogy(fd, Pxxd, color = 'red', label = 'EMD Audio Signal', alpha = 0.5)
    plt.title(f'Power Spectral Density | RMSE: {rmse[2]:.4f} | SNR: {snr[2]:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout(pad=3.0) 
    plt.show()

# ------------------------------------------------------------- MAIN FILE ----------------------------------------------------------------------------

def main():
    
    # countdown
    show_countdown()
    duration = 3
    
    # record the audio
    audio_data, sr = capture_audio(duration)
    
    # time axis
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Filter using Butter Worth BandPass Filter
    butter_filtered_signal = butterworth_filter(audio_data, LOWCUT, HIGHCUT, ORDER, sr)
    print("-----------  ButterWorth Filter Stage Clear -----------------\n")
    
    # # Fiter using ICEEMDAN Filter on raww
    # _denoised_signal, _imf0 = iceemdan_filter(audio_data)
    # denoised_signal = wavelet_denoising(_denoised_signal)
    
    # # Fiter using ICEEMDAN Filter on butter worth filter audio signal
    # B_denoised_signal, B_imf0 = iceemdan_filter(butter_filtered_signal)
    # B_denoised_signal = wavelet_denoising(B_denoised_signal)
    # print("-------------  ICEEMDAN Filter Stage Clear  --------------------\n")
    
    # Fiter using ICEEMDAN Filter on raww
    _ice_denoised, ice_imf0 = iceemdan_filter(butter_filtered_signal)
    ice_denoised = wavelet_denoising(_ice_denoised)
    
    # Fiter using ICEEMDAN Filter on butter worth filter audio signal
    _denoised_signal, B_imf0 = emd_filter(butter_filtered_signal)
    B_denoised_signal = wavelet_denoising(_denoised_signal)
    print("-------------  ICEEMDAN Filter Stage Clear  --------------------\n")
    
    # Calculat the SNR and RMSE on audio signal 
    fitered_signal = [butter_filtered_signal, ice_denoised, B_denoised_signal]
    # fiters = [butter_filtered_signal, denoised_signal, B_denoised_signal]
    A_snrs = [calculate_snr(audio_data, signal) for signal in fitered_signal]
    A_rmses = [calculate_rmse(audio_data, signal) for signal in fitered_signal]
    
    
    # Calculate the SNR and RMSE for butter worth filter
    snrs = [calculate_snr(butter_filtered_signal, signal) for signal in fitered_signal]
    rmses = [calculate_rmse(butter_filtered_signal, signal) for signal in fitered_signal]
    print(" -------------------  Here We GO.. -----------------------------\n")
    
    # plots 
    plot_signal(time, audio_data, butter_filtered_signal, ice_denoised, B_denoised_signal, sr, A_rmses, A_snrs)
    plot_psd(audio_data, butter_filtered_signal,butter_filtered_signal, ice_denoised, B_denoised_signal, sr, rmses, snrs, A_rmses,A_snrs)
    
    print(f' SNR BUTTER:{snrs[0]:.2f} \n SNR IMF0: {snrs[1]:.2f} \n SNR Denoised: {snrs[2]:.2f} \n')
    print(f' RMSE BUTTER:{rmses[0]:.4f} \n RMSE IMF0: {rmses[1]:.4f} \n RMSE Denoised: {rmses[2]:.4f} \n')
    

if __name__ == '__main__':
    # Allow user to retry if desired
    again = True
    while again:
        main()
        ans = input("Try Again? y/n: ").lower()
        if ans == 'n' or ans =='no':
            again = False
        elif ans == 'yes' or ans == 'y':
            again = True
        else:
            print("Invalid Response!!")