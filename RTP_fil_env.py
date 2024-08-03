#***************** Import Dependencies ***************** 

from scipy.signal import kaiserord, firwin, filtfilt
from scipy.signal import hilbert, butter, lfilter
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np


# **************** Processing real time audio **********


## parameters
SAMPLE_RATE = 4000   # samples per second
BUFFER_SIZE = 20000     # Buffer size
INTERVAL = 20           # interval of 20ms 
WINDOW_SIZE = 20        # smoothen window size
ORDER = 5               #
LOWCUT = 100
HIGHCUT = 1000
FL_HZ = 10
RIPPLE_DB = 10.0


## Initialized the plot 
fig, ax = plt.subplots()
x = np.arange(0, BUFFER_SIZE) / SAMPLE_RATE
line_waveform, = ax.plot(x, np.zeros(BUFFER_SIZE), label="Waveform")
line_envelope, = ax.plot(x, np.zeros(BUFFER_SIZE), label="Envelope", color='r',alpha = 0.8)
ax.set_ylim(-1, 1)
ax.set_xlim(0, BUFFER_SIZE / SAMPLE_RATE)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Real-Time Audio Waveform and Smoothened Envelope")
ax.legend(loc="upper right")


## Countdown text
countdown_text = ax.text(0.5, 0.5, '', transform=ax.transAxes, fontsize=20, va='center', ha='center')


## Global variable to store the latest audio buffer
latest_audio_buffer = np.zeros(BUFFER_SIZE)


## continuous audio streaming
def audio_callback(in_data, frame_count, time_info, status):
    '''
    Objective: Update the latest audio signal data into the buffer.
    
    Input:
    - in_data: ndarray, incoming audio data
    - frame_count: int, number of frames per buffer
    - time_info: dict, timing information
    - status: CData, callback status indicating errors or warnings
    
    Output: None
    '''
    
    global latest_audio_buffer
    if status:
        print(status)
    latest_audio_buffer = in_data[:, 0]

    
## Function to smoothen the envelope
def smooth_envelope(signal, window_size):
    '''
    Objective: Smooth the envelope of the signal using a moving average filter.
    
    Input:
    - signal: ndarray, the input signal to be smoothed
    - window_size: int, the size of the moving average window
    
    Output:
    - ndarray, the smoothed signal
    '''
    
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


## Create a stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)


## Butterworth filter on signal to filter the signal
def butter_bp(lowcut, highcut, fs, order):
    '''
    Objective: Design a Butterworth bandpass filter.
    
    Input:
    - lowcut: float, lower cutoff frequency
    - highcut: float, upper cutoff frequency
    - fs: int, sampling rate
    - order: int, order of the filter
    
    Output:
    - b, a: ndarray, filter coefficients
    '''
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
    
## Filtering the signal
def filter_signal(signal, lowcut, highcut, order, sr):
    '''
    Objective: Apply a Butterworth bandpass filter to the signal.
    
    Input:
    - signal: ndarray, input signal to be filtered
    - lowcut: float, lower cutoff frequency
    - highcut: float, upper cutoff frequency
    - order: int, order of the filter
    - sr: int, sampling rate
    
    Output:
    - y: ndarray, filtered signal
    '''
    
    b, a = butter_bp(lowcut, highcut, sr, order)
    y = lfilter(b, a, signal)
    return y


## Creating envelope 
def create_envelope(signal, fs, fL_hz=FL_HZ, ripple_db=RIPPLE_DB):
    '''
    Objective: Create and filter the Hilbert envelope of the signal.
    
    Input:
    - signal: ndarray, input signal
    - fs: int, sampling rate
    - fL_hz: float, lowpass filter cutoff frequency
    - ripple_db: float, maximum ripple allowed in the passband
    
    Output:
    - filtered_envelope: ndarray, filtered envelope of the signal
    '''
    
    nyq_rate = 0.5 * fs
    width = 1.0 / nyq_rate

    ### Hilbert transform
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    # Filter Hilbert envelope
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, fL_hz / nyq_rate, window=('kaiser', beta))
    filtered_envelope = filtfilt(taps, 1, envelope)
    
    return filtered_envelope
    
    
## Animate update function
def update(frame):
    '''
    Objective: Update the plot with the latest audio buffer and its envelope.
    
    Input:
    - frame: int, current frame number
    
    Output:
    - line_waveform: Line2D object, updated waveform line
    - line_envelope: Line2D object, updated envelope line
    '''
    
    global latest_audio_buffer
    line_waveform.set_ydata(latest_audio_buffer)
    
    ## Filter signal
    fil_signal = filter_signal(latest_audio_buffer, lowcut=LOWCUT, highcut=HIGHCUT, order=ORDER, sr=SAMPLE_RATE)
    envelope = create_envelope(fil_signal, SAMPLE_RATE, fL_hz= FL_HZ, ripple_db= RIPPLE_DB)
    
    ### Normalize the audio and calculate the Hilbert transform
    # max_amp = np.max(np.abs(envelope))
    # normalized_env = envelope / max_amp
    
    ### Smooth envelope
    smoothen_envelope = smooth_envelope(envelope, WINDOW_SIZE)
    line_envelope.set_ydata(smoothen_envelope)
    return line_waveform, line_envelope


## Animation update function
ani = FuncAnimation(fig, update, blit=True, interval=INTERVAL, cache_frame_data=False)


## Countdown function
def countdown(duration):
    
    for i in range(duration, 0, -1):
        countdown_text.set_text(f"Inhale in {i}...")
        plt.pause(1)
    countdown_text.set_text("Go!")
    plt.pause(1)
    countdown_text.set_text("")


## Start the stream and show the plot
with stream:
    # countdown(5)
    plt.show()
