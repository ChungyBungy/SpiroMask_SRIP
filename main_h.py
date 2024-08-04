'''
Data Collection Information:

Name:                                       Gender:
Body Type : 
    - A: Tall and Healthy
    - B: Tall and Skinny  
    - C: Short and Healthy
    - D: Short and Skinny 
Signal:
Filename :  recorded wav file
testing time and date:
Max Peak:
Peak at 1 sec:
Signal Status

'''

# Import the Dependencies
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.signal import find_peaks
from scipy.io.wavfile import write
from PyEMD import EMD
import numpy as np
import tkinter as tk
import pywt
from time import sleep
import datetime
import os
from pathlib import Path
import csv
from scipy.integrate import cumtrapz

# -----------------------------   GLOBAL PARAMETERS   ---------------------------------------
SAMPLE_RATE = 8000
LOWCUT = 100
HIGHCUT = 2000
ORDER = 5
WINDOW_SIZE = 1000
FL_HZ = 10
RIPPLE_DB = 10.0
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------   RECORD AUDIO SIGNAL   ------------------------------------
def capture_audio(duration: int):
    print('Recording....')
    audio_data = sd.rec(int(duration * SAMPLE_RATE), SAMPLE_RATE, channels=1, dtype='int16')
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
    countdown_window.attributes('-topmost', True)
    countdown_label = tk.Label(countdown_window, text="", font=("Arial", 24))
    countdown_label.pack()

    for i in range(5, 0, -1):
        countdown_label.config(text=f"Get ready!\nStarting in {i}...")
        countdown_window.update()
        sleep(1)

    countdown_window.destroy()

# --------------------------------------------------   BUTTERWORTH BANDPASS (100 - 1000)    ----------------------------------------------
def butter_bp(lowcut: int, highcut: int, sr: int, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ss.butter(order, [low, high], btype='band')
    return b, a

def butterworth_filter(signal: list, lowcut: int, highcut: int, order: int, sr: int):
    b, a = butter_bp(lowcut, highcut, sr, order)
    y = ss.lfilter(b, a, signal)
    return y

# -------------------------------------------------------------  EMD FILTER  ---------------------------------------------------------
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

# ---------------------------------------------------------------- EVALUATION ------------------------------------------------------------------
def calculate_snr(signal: list, denoise_signal: list) -> float:
    noise = signal - denoise_signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_rmse(signal: list, denoise_signal: list) -> float:
    rmse = np.sqrt(np.mean((signal - denoise_signal)** 2))
    return rmse

# ------------------------------------------------------------ EXTRACT ENVELOPE AND SMOOTHEN IT   ---------------------------------------------------------------
def extract_envelope(signal: list, sr: int, fL_hz: int, ripple_db: int) -> list:
    nyq_rate = 0.5 * sr
    width = 1.0 / nyq_rate
    
    analytic_signal = ss.hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    N, beta = ss.kaiserord(ripple_db, width)
    taps = ss.firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
    filtered_envelope = ss.filtfilt(taps, 1, envelope)
    return filtered_envelope

def smooth_envelope(signal: list, window_size: int) -> list:
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# --------------------------------------------------------------  CALCULATE PARAMETERS ----------------------------------------------------
def maxima(signal: list) -> tuple:
    signal = np.array(signal)  
    index = np.argmax(signal)  
    maxx_signal = signal[index]
    return index, maxx_signal

#  -------------------------------------------------------------- PLOTS FUNCTION --------------------------------------------------------------------
# def plot_signals(time: list, sr: int, signal: list, denoised_signal: list, envelope: list, maxx_peak_time: int, start: int, end: int, name: str,testing_datetime: str) -> None:
    
#     plt.figure(figsize=(14, 10))
#     plt.subplot(3, 1, 1)
#     plt.plot(time, signal, label='Original Signal')
#     plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
#     plt.title('Original V/s Denoised Signal')
#     plt.xlabel('Time(s)')
#     plt.ylabel('Amplitude')
#     plt.legend(loc='upper right')
#     plt.grid(True)
    
#     plt.subplot(3, 1, 2)
#     plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
#     plt.plot(time, envelope, label='Envelope Signal', color='r', alpha=0.7)
#     plt.plot(time[sr], envelope[sr], 'go', label='Peak at 1 sec')
#     plt.plot(time[maxx_peak_time], envelope[maxx_peak_time], 'mo', label='Max Peak')
#     plt.plot(time[start], envelope[start], 'ko', label='Start Point')
#     plt.plot(time[end], envelope[end], 'co', label='End Point')
#     plt.title(f'Max Peak: {envelope[maxx_peak_time]:.2f} | Peak at 1 sec: {envelope[sr]:.2f}')
#     plt.xlabel('Time(s)')
#     plt.ylabel('Amplitude')
#     plt.legend(loc='upper right')
#     plt.grid(True)
    
#     clipped_envelope = envelope[start:end]

#     plt.subplot(3, 1, 3)
#     plt.plot(time, clipped_envelope, label='Envelope Signal', color='m', alpha=0.7)
#     plt.title('Extracted Envelope')
#     plt.xlabel('Time(s)')
#     plt.ylabel('Amplitude')
#     plt.legend(loc='upper right')
#     plt.grid(True)
    
#     plt.tight_layout(pad=3.0)
    
#     filename3 = (f"{name}_{testing_datetime}.jpg")
#     data_file = BASE_DIR / 'Data_Collection' / 'Plots'
#     file_path = os.path.join(data_file,filename3)
#     plt.savefig(file_path)
    
#     plt.show()

BASE_DIR = Path(__file__).resolve().parent

def plot_signals(time: list, sr: int, signal: list, denoised_signal: list, envelope: list, maxx_peak_time: int, start: int, end: int, name: str, testing_datetime: str) -> None:
    
    plt.figure(figsize=(14, 10))
    
    # Plot Original and Denoised Signal
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Original Signal')
    plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.title('Original V/s Denoised Signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    clipped_envelope = envelope[start:end]

    # Plot Denoised Signal and Envelope
    plt.subplot(3, 1, 2)
    plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.plot(time, envelope, label='Envelope Signal', color='r', alpha=0.7)
    plt.plot(time[sr], envelope[sr], 'go', label='Peak at 1 sec')
    plt.plot(time[maxx_peak_time], envelope[maxx_peak_time], 'mo', label='Max Peak')
    plt.plot(time[start], envelope[start], 'ko', label='Start Point')
    plt.plot(time[end], envelope[end], 'co', label='End Point')
    plt.title(f'Max Peak: {envelope[maxx_peak_time]:.2f} | Peak at 1 sec: {clipped_envelope[sr]:.2f}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Clip the envelope and corresponding time
    clipped_time = time[start:end]
    
    volume = cumtrapz(clipped_envelope, clipped_time, initial=0)

    # Plot Extracted Envelope
    plt.subplot(4, 1, 4)
    plt.plot(clipped_time, volume, label='Volume-Time Graph', color='b', alpha=0.7)
    plt.title('Volume-Time Graph')
    plt.xlabel('Time(s)')
    plt.ylabel('Volume')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    filename3 = f"{name}_{testing_datetime}.jpg"
    data_file = BASE_DIR / 'Data_Collection2' / 'Plots'
    os.makedirs(data_file, exist_ok=True)
    file_path = os.path.join(data_file, filename3)
    plt.savefig(file_path)
    
    plt.show()
    
    
# -------------------------------------------------- CLACULATING FVC ---------------------------------------------------------------

def get_fvc(signal : list, time : list , start: int, end: int):
    clipped_time = time[start:end]
    volume = cumtrapz(signal, clipped_time, initial=0)
    fvc = np.max(volume)
    return fvc
  
# ------------------------------------------------------------- CLIPPING AUDIO  ------------------------------------------------------------------------
def clipping_audio(signal: list, window_size: int, step_size: int, threshold_ratio: float) -> tuple:
    start = 0
    endpoint = 0
    signal = np.array(signal)
    maxx = np.max(signal) ** 2
    threshold = threshold_ratio * maxx
    for i in range(0, len(signal) - window_size + 1, step_size):
        energy = np.sum((signal[i: i + window_size]) ** 2)
        if start == 0 and energy > threshold:
            start = i
        if start != 0 and energy < threshold:
            endpoint = i 
            break
    
    return start, endpoint

# -------------------------------------------------- RECORDS THE DATA COLLECTION FUNCTION ---------------------------------------------------------------
def record_data(filepath: str, subject_number: int, name: str, body_type: int, gender: str, diseases: str) -> None:
    # Ensure the folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the file in append mode and write the data
    with open(filepath, 'a') as file:
        # If the file is empty, write the header
        if os.path.getsize(filepath) == 0:
            file.write("subject_number,name,body_type,gender,diseases\n")
        
        # Write the data
        file.write(f'\n{subject_number},{name},{body_type},{gender},{diseases}')
    print(f"Data appended to '{filepath}' successfully.")

def collect_data():
    print('\n<--------------------------------------  Subject Details  ------------------------------------------>\n')
    subject_number = int(input('Subject Number: '))
    name = str(input('Subject Full Name: ')).lower().strip().replace(' ', '_')
    gender = str(input('Subject Gender: ')).lower().strip()
    body_type = str(input('Subject Body Type[A, B, C, D]: ')).upper().strip()
    diseases = str(input('Any Lung Related Diseases?? Yes/No: ')).lower()
    print('\n<--------------------------------------  Details Collected  ------------------------------------------>\n')
    return subject_number, name, body_type, gender, diseases

def save_file(audio_data: np.ndarray, sr: int, filename: str, dir: str) -> None:
    filepath = os.path.join(dir, filename)
    write(filepath, sr, audio_data)
    print(f"Audio data saved to {filepath}")


def record_data2(filepath: str, recorded_audio_filename: str, subject_number: int, testing_datetime: str, max_peak: float, peak_at_1sec: float, volume: str) -> None:
    # Ensure the folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the file in append mode and write the data
    with open(filepath, 'a') as file:
        # If the file is empty, write the header
        if os.path.getsize(filepath) == 0:
            file.write("subject_number,testing_datetime,max_peak,peak_at_1sec,FVC\n")
            #file.write("filename,subject_number,signal,testing_datetime,max_peak,peak_at_1sec,signal_status\n")
        
        # Write the data
        file.write(f'\n{subject_number},{testing_datetime},{max_peak},{peak_at_1sec},{volume}')
    print(f"Data appended to '{filepath}' successfully.")

def get_name(file_path, row_index):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i == (row_index+1):
                return row[1]

        
# -------------------------------------------------------------------------- MAIN FUNCTION  ----------------------------------------------------------------------------
# -------------------------------------------------------------------------- MAIN FUNCTION  ----------------------------------------------------------------------------
def main():
    print("--------------  Initiate System  ------------------")
    sleep(1)
    
    data_file = BASE_DIR / 'Data_Collection2' / 'Users.csv'
    
    # Ask user 
    answ = str(input("Is this your first time in the system ? : ")).lower()
    if answ in ['yes', 'y']:
        subject_number, name, body_type, gender, diseases = collect_data()
        record_data(data_file, subject_number, name, body_type, gender, diseases)
    else:
        user_id = int(input("Enter your user id: "))
        name = str(get_name(data_file, user_id))
        
    # Collect subject data
    sleep(1)
    
    # Countdown
    show_countdown()

    # Record the audio
    current_day = datetime.datetime.now()
    testing_datetime = current_day.strftime("%Y%m%d_%H%M%S")
    
    data_file2 = BASE_DIR / 'Data_Collection2' / 'records.csv'
    audio_data, sr = capture_audio(3)
    print(*audio_data[:100], sep='\n')
    
    # Time axis
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Apply ButterWorth Pass Signal
    butter_filt_data = butterworth_filter(audio_data, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
    
    # Apply EMD filter on that
    denoised_signal, imf0 = emd_filter(butter_filt_data)
    denoised_signal = wavelet_denoising(denoised_signal)
    
    # Extracting Envelope from the filtered audio
    envelope = extract_envelope(denoised_signal, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
    smoothen_envelope = smooth_envelope(envelope, WINDOW_SIZE)
        
    # Find the timestamp where the amplitude is maximum
    max_peak_time, maxx_peak = maxima(smoothen_envelope)
    maxx_peak = float(f"{maxx_peak:.2f}")
    
    peak_at_1s = smoothen_envelope[sr]
    peak_at_1s = float(f"{peak_at_1s:.2f}")
    
    # Find the clips in the audio signal
    start_point, end_point = clipping_audio(envelope, 400, 1, 0.1)

    clipped_time = time[start_point:end_point]

    # Clip the smoothen_envelope to match the length of clipped_time
    clipped_envelope = smoothen_envelope[start_point:end_point]

    # Calculate volume using cumtrapz
    volume = cumtrapz(clipped_envelope, clipped_time, initial=0)
    
    
    # Plot the signal
    plot_signals(time, sr, audio_data, denoised_signal, smoothen_envelope, max_peak_time, start_point, end_point, name, testing_datetime)

    
    FVC = round(np.max(volume),2)
    FEV1 = round(clipped_envelope[sr],2)
    PEF_pre = maxima(smoothen_envelope)
    PEF = round(PEF_pre[0],2)

    
    
    dir1 = BASE_DIR / 'Data_Collection2' / 'collected_audios' / 'FEV1'
    dir2 = BASE_DIR / 'Data_Collection2' / 'collected_audios' / 'FVC'
    dir3 = BASE_DIR / 'Data_Collection2' / 'collected_audios' / 'PEF'

    filename = f'{name}_{testing_datetime}.wav'

    
    filename1 = f'{FEV1}.wav'
    filename2 = f'{FVC}.wav'
    filename3 = f'{PEF}.wav'
    
    print(FVC)

    
    save_file(audio_data, sr, filename1, dir1)
    save_file(audio_data, sr, filename2, dir2)
    save_file(audio_data, sr, filename3, dir3)

    if answ in ['yes', 'y']:
        record_data2(data_file2, filename, subject_number, testing_datetime, maxx_peak, peak_at_1s, FVC)
    else:
        record_data2(data_file2, filename, user_id, testing_datetime, maxx_peak, peak_at_1s, FVC)

if __name__ == '__main__':
    again = True
    while again:
        main()
        ans = input("Try Again? y/n: ").lower()
        if ans in ['n', 'no']:
            again = False
        elif ans in ['yes', 'y', '1']:
            again = True
        else:
            print("Invalid Response!!")

