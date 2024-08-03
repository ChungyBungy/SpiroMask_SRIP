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

# ---------------------------------------------------  Import the Dependencies  ---------------------------------------------------------
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.io.wavfile import write
from PyEMD import EMD
import numpy as np
import tkinter as tk
import pywt
from time import sleep
import datetime
import os
from pathlib import Path
from scipy.integrate import cumtrapz
# -------------------------------------------------------   GLOBAL PARAMETERS   ---------------------------------------------------------
SAMPLE_RATE = 4000
LOWCUT = 100
HIGHCUT = 1000
ORDER = 5
WINDOW_SIZE = 400
FL_HZ = 10
RIPPLE_DB = 10.0
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------   RECORD AUDIO SIGNAL   ------------------------------------------------------

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
    countdown_label = tk.Label(countdown_window, text="", font=("Arial", 24))
    countdown_label.pack()

    for i in range(5, 0, -1):
        countdown_label.config(text=f"Get ready!\nStarting in {i}...")
        countdown_window.update()
        sleep(1)

    countdown_window.destroy()

# --------------------------------------------------   BUTTERWORTH BANDPASS (100 - 1000)    ---------------------------------------------------

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

# -------------------------------------------------------------  EMD FILTER  --------------------------------------------------------------------

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

# ----------------------------------------------------------------- EVALUATION --------------------------------------------------------------------------------

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

# --------------------------------------------------------------  CALCULATE MAXIMA ----------------------------------------------------

def maxima(signal: list) -> tuple:
    signal = np.array(signal)  
    index = np.argmax(signal)  
    return index

# ------------------------------------------------------------ PLOT REJECTION -------------------------------------------------------------

def plot_rejection(time: list, sr: int, signal: list, denoised_signal: list, envelope: list, maxx_peak_time: int, peak_at_1sec_time:int ,start: int, end: int, reason: str) -> None:
    
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Original Signal')
    # plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.title(f'XXXXXX SAMPLE REJECTED!! XXXXXXX Reason: {reason}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.plot(time, envelope, label='Envelope Signal', color='r', alpha=0.7)
    plt.plot(time[peak_at_1sec_time], envelope[peak_at_1sec_time], 'go', label='Peak at 1 sec')
    plt.plot(time[maxx_peak_time], envelope[maxx_peak_time], 'mo', label='Max Peak')
    plt.plot(time[start], envelope[start], 'ko', label='Start Point')
    plt.plot(time[end], envelope[end], 'co', label='End Point')
    plt.title(f'Max Peak: {envelope[maxx_peak_time]:.2f} | Peak at 1 sec: {envelope[peak_at_1sec_time]:.2f}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, envelope, label='Envelope Signal', color='m', alpha=0.7)
    plt.title('Extracted Envelope')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.show()

#  -------------------------------------------------------------- PLOTS ACCEPTANCE FUNCTION --------------------------------------------------------------------

def plot_signals(time: list, sr: int, signal: list, denoised_signal: list, envelope: list, maxx_peak_time: int, peak_at_1sec_time:int ,start: int, end: int, reason: str) -> None:
    
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Original Signal')
    # plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.title(f'Original V/s Denoised Signal || SNR :{reason}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, denoised_signal, label='Denoised Signal', color='g', alpha=0.5)
    plt.plot(time, envelope, label='Envelope Signal', color='r', alpha=0.7)
    plt.plot(time[peak_at_1sec_time], envelope[peak_at_1sec_time], 'go', label='Peak at 1 sec')
    plt.plot(time[maxx_peak_time], envelope[maxx_peak_time], 'mo', label='Max Peak')
    plt.plot(time[start], envelope[start], 'ko', label='Start Point')
    plt.plot(time[end], envelope[end], 'co', label='End Point')
    plt.title(f'Max Peak: {envelope[maxx_peak_time]:.2f} | Peak at 1 sec: {envelope[peak_at_1sec_time]:.2f}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, envelope, label='Envelope Signal', color='m', alpha=0.7)
    plt.title('Extracted Envelope')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
# ------------------------------------------------------------- CLIPPING AUDIO  ------------------------------------------------------------------------

# def clipping_audio(signal: list, window_size: int, step_size: int, threshold_ratio: float) -> tuple:
#     start = 0
#     endpoint = 0
#     signal = np.array(signal)
#     maxx = np.max(signal) ** 2
#     threshold = threshold_ratio * maxx
#     for i in range(0, len(signal) - window_size + 1, step_size):
#         energy = np.sum((signal[i: i + window_size]) ** 2)
#         if start == 0 and energy > threshold:
#             start = i
#         if start != 0 and energy < threshold:
#             endpoint = i 
#             break
    
#     return start, endpoint

def clipping_audio(signal: list, window_size: int, step_size: int, threshold_ratio: float) -> tuple:
    START, END = 0, 0
    start, endpoint = 0, 0
    prev_energy = 0
    signal = np.array(signal)
    maxx = np.max(signal) ** 2
    threshold = threshold_ratio * maxx
    
    # clipping the waveform based on their energy level and clipped only that section that have maximum energy
    for i in range(0, len(signal) - window_size + 1, step_size):
        energy = np.sum((signal[i: i + window_size]) ** 2)
        if start == 0 and energy > 4 * threshold:
            start = i
        if start != 0 and energy < threshold:
            endpoint = i 
            curr_clip_energy = np.sum(signal[start : endpoint + 1] **2 )
            if curr_clip_energy >  prev_energy:
                START, END = start, endpoint
                prev_energy = curr_clip_energy
            start = 0
            endpoint = 0
    
    # return the clipped section with maximum energy
    return START, END

# ------------------------------------------------------------- ENVELOPE SMOOTHNESS ESTIMATION  ------------------------------------------------------------------------
    
def disrupt_detection(signal: np.ndarray, threshold: float):
    std_deviation = np.std(signal)
    abrupt_indices = np.where(np.abs(signal - np.mean(signal)) > threshold * std_deviation)[0]
    return abrupt_indices

# ------------------------------------------------------------- REJECTION / ACCEPTANCE  ------------------------------------------------------------------------

def isAccept(signal: np.ndarray, start: int, end: int, snr_threshold: float):
    '''
    Accepted Signal must follow following conditions:
    1. Have only one maxia
    2. SNR
    '''
    
    # clipped signal
    clipped_signal = signal[start : end + 1]
    
    # max peak 
    max_peak = np.max(clipped_signal)
    y_threshold = 0.5 * max_peak
    x_threshold = 0.4 * (end - start + 1)
    
    # signal should only have one max peak
    peaks, _ = ss.find_peaks(clipped_signal, height=y_threshold, distance=x_threshold)
    
    noise_std = np.std(clipped_signal)
    signal_power = np.mean(clipped_signal**2)
    noise_power = np.mean(noise_std**2)
    snr = 10 * np.log(signal_power / noise_power)
    
    # condition 1: Peaks should not be greater than one
    if len(peaks) > 1:
        return False, f'More than one peaks: {len(peaks)}'
    
    # condition 2: SNR > threshold_snr
    if snr < snr_threshold:
        return False, f"Couldn't Satisfied SNR condition, SNR :{snr:.2f}"
    
    return True, f'{snr:.2f}'

# -------------------------------------------------- RECORDS THE DATA COLLECTION FUNCTION ---------------------------------------------------------------

def record_data(filepath: str, recorded_audio_filename: str, subject_number: int, name: str, body_type: int, gender: str, testing_datetime: str, diseases: str, max_peak: float, peak_at_1sec: float, signal_status: str) -> None:
    # Ensure the folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the file in append mode and write the data
    with open(filepath, 'a') as file:
        # If the file is empty, write the header
        if os.path.getsize(filepath) == 0:
            file.write("filename,subject_number,name,signal,body_type,gender,testing_datetime,diseases,max_peak,peak_at_1sec,signal_status\n")
        
        # Write the data
        file.write(f'\n{recorded_audio_filename},{subject_number},{name},{body_type},{gender},{testing_datetime},{diseases},{max_peak},{peak_at_1sec},{signal_status}')
    print(f"Data appended to '{filepath}' successfully.")

def collect_data():
    print('\n<--------------------------------------  Subject Details  ------------------------------------------>\n')
    subject_number = int(input('Subject Number: '))
    name = str(input('Subject Full Name: ')).lower().strip().replace(' ', '_')
    gender = str(input('Subject Gender: ')).lower().strip()
    body_type = str(input('Subject Body Type[A, B, C, D]: ')).upper().strip()
    diseases = str(input('Any Heart Related Diseases?? Yes/No: ')).lower()
    print('\n<--------------------------------------  Details Collected  ------------------------------------------>\n')
    return subject_number, name, body_type, gender, diseases

def save_file(audio_data: np.ndarray, sr: int, filename: str, dir: str) -> None:
    filepath = os.path.join(dir, filename)
    write(filepath, sr, audio_data)
    print(f"Audio data saved to {filepath}")

# -------------------------------------------------------------------------- MAIN FUNCTION  ----------------------------------------------------------------------------
def main(subject_number:int, name: str, body_type:str, gender:str, diseases: str):
    # Countdown
    show_countdown()

    # Record the audio
    current_day = datetime.datetime.now()
    testing_datetime = current_day.strftime("%Y%m%d_%H%M%S")

    data_file = BASE_DIR / 'Data_Collection' / 'records_.csv'
    audio_data, sr = capture_audio(3)
    
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

    # # Find the clips in the audio signal
    start_point, end_point = clipping_audio(smoothen_envelope, 400, 1, 0.2)
    clipped_time = time[start_point:end_point+ 1]
    clipped_audio = smoothen_envelope[start_point: end_point +1]
    
    # Calculate FVC
    volume = cumtrapz(clipped_audio, clipped_time, initial = 0)
    FVC = round(np.max(volume),2)
    
    # Find the timestamp where the amplitude is maximum
    max_peak_time = maxima(smoothen_envelope[start_point : end_point + 1])
    max_peak_time = max_peak_time + start_point
    maxx_peak = f'{smoothen_envelope[max_peak_time]:.2f}'
    
    peak_at_1s_time = start_point + sr
    try:
        peak_at_1s = smoothen_envelope[peak_at_1s_time]
        peak_at_1s = float(f"{peak_at_1s:.2f}")
    except IndexError:
        peak_at_1s = 'Appropriate Exhale, Try to Exhale Earlier when recording starts'
            
    filename = f'{name}_{testing_datetime}.wav'
    print(f"Save the plot as: {name}_{testing_datetime}")
    
    # check for acceptance of the signal
    accept, reason = isAccept(envelope,start_point, end_point, 10.0)
    
    # Plot the signal
    if accept:
        plot_signals(time, sr, audio_data, denoised_signal, smoothen_envelope, max_peak_time, peak_at_1s_time, start_point, end_point, reason)
    else:
        plot_rejection(time, sr, audio_data, denoised_signal, smoothen_envelope, max_peak_time, peak_at_1s_time, start_point, end_point, reason)
        
    # Feedback on this 
    signal_status = str(input("Accepted or not?? Yes or No: ")).lower()
    
    # if signal_status == 'yes':
    #     dir = BASE_DIR / 'Data_Collection' / 'collected_audios' / 'Accepted'
    # else:
    #     dir = BASE_DIR / 'Data_Collection' / 'collected_audios' / 'Rejected'
    # dir.mkdir(parents=True, exist_ok=True)
    
    # save_file(audio_data, sr, filename, dir)
    # record_data(data_file, filename, subject_number, name, body_type, gender, testing_datetime, diseases, maxx_peak, peak_at_1s, signal_status)

if __name__ == '__main__':
    print("--------------  Initiate System  ------------------")
    sleep(1)
    subject_number, name, body_type, gender, diseases = collect_data()
    again = True
    while again:
        main(subject_number, name, body_type, gender, diseases)
        # main()
        ans = input("Try Again? y/n: ").lower()
        if ans in ['n', 'no']:
            again = False
        elif ans in ['yes', 'y']:
            again = True
        else:
            print("Invalid Response!!")
            break
            