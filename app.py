## TODO:
'''
Force Vital Capacity Estimating Fuction Integration is not done yet.
well, its container is already made, just code it and have to replce FVC value in there
'''

import sounddevice as sd
import scipy.signal as ss
import numpy as np
from flet import *
from time import sleep
from PyEMD import EMD
import pywt
from scipy.integrate import cumulative_trapezoid as cumtrapz

# Constants
SAMPLE_RATE = 4000
DURATION = 3  # seconds
LOWCUT = 100
HIGHCUT = 1000
ORDER = 5
WINDOW_SIZE = 800
FL_HZ = 10
RIPPLE_DB = 10.0

# Function to record audio
def record_audio(DURATION: int):
    print("Recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data.flatten(), SAMPLE_RATE

# Function to apply Butterworth filter
def butterworth_filter(signal, lowcut, highcut, order, sr):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = ss.butter(order, [low, high], btype='band')
    return ss.lfilter(b, a, signal)

# Function to apply EMD filter
def emd_filter(signal):
    emd = EMD()
    imfs = emd.emd(signal)
    denoised_signal = signal - imfs[0]
    return denoised_signal

# Function to perform wavelet denoising
def wavelet_denoising(signal):
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, 'db1')

# Function to extract envelope
def extract_envelope(signal, sr, fl_hz, ripple_db):
    nyq_rate = 0.5 * sr
    width = 1.0 / nyq_rate
    analytic_signal = ss.hilbert(signal)
    envelope = np.abs(analytic_signal)
    n, beta = ss.kaiserord(ripple_db, width)
    taps = ss.firwin(n, fl_hz / nyq_rate, window=('kaiser', beta))
    return ss.filtfilt(taps, 1, envelope)

# Function to smooth envelope
def smooth_envelope(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to find maxima
def maxima(signal):
    index = np.argmax(signal)
    return signal[index]

# Clipping audio
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

# Main function to build the app
def main(page: Page):
    page.title = "SpiroMask.ai"
    page.horizontal_alignment = CrossAxisAlignment.CENTER
    page.vertical_alignment = MainAxisAlignment.CENTER

    # Add a navigation bar with icons
    def nav_bar():
        return Container(
            content=Row(
                controls=[
                    IconButton(icon=icons.HOME, on_click=lambda e: page.go("/")),
                    IconButton(icon=icons.MEDICAL_SERVICES, on_click=lambda e: page.go("/record")),
                ],
                alignment=MainAxisAlignment.SPACE_AROUND,
            ),
            alignment=alignment.bottom_center,
            padding=padding.all(10),
            margin=margin.only(top=10),  # Ensure space between content and nav bar
        )

    status_text = Text("SpiroMask.ai", size=30, weight="bold", text_align=TextAlign.CENTER)
    start_button = ElevatedButton("Start", on_click=lambda e: record_and_process(), width=120, height=50)
    # Instruction text
    instructions_text = Text(
        "INSTRUCTIONS:\n\n"
        "1. Connect your earbuds via Bluetooth.\n"
        "2. Place the earbuds (mic) into your N95 mask at nostril level.\n"
        "3. Sit comfortably with a straight posture.\n"
        "4. Click 'Start' to begin. Follow the 5-second countdown:\n"
        "   - Inhale for 3 seconds.\n"
        "   - Hold for 2 seconds.\n"
        "   - Exhale forcefully when the countdown ends.\n",
        
        size=16,
        text_align=TextAlign.LEFT,
        weight=FontWeight.NORMAL,
        color=colors.WHITE
    )
    # start_button1 = ElevatedButton("Start", on_click=lambda e: record_and_process(), width=150, height=50)
    
    
    def show_countdown():
        start_button.visible = False
        status_text.page.padding = padding.only(top=200)
        instructions_text.visible = False
        page.update()
        for i in range(5, 0, -1):
            status_text.value = f"GET READY!! Starting in {i}..."
            
            page.update()
            sleep(1)
        status_text.value = "Recording..."
        page.update()

    def record_and_process():
        show_countdown()
        status_text.page.padding = padding.only(top=30)
        audio_data, sample_rate = record_audio(DURATION)
        sample_indices = np.arange(len(audio_data))
        
        # Apply the Butterworth Filter
        bw_filter = butterworth_filter(audio_data, LOWCUT, HIGHCUT, ORDER, SAMPLE_RATE)
        
        # EMD Filter
        denoised_signal = emd_filter(bw_filter)
        filtered_audio = wavelet_denoising(denoised_signal)
        
        # Extract the envelope
        envelope = extract_envelope(filtered_audio, SAMPLE_RATE, FL_HZ, RIPPLE_DB)
        smoothen_envelope = smooth_envelope(envelope, WINDOW_SIZE)
        
        # # Find the clips in the audio signal
        start_point, end_point = clipping_audio(smoothen_envelope, 400, 1, 0.2)
        clipped_time = sample_indices[start_point:end_point+ 1]
        clipped_audio = smoothen_envelope[start_point: end_point +1]
        
        # Calculate FVC
        volume = cumtrapz(clipped_audio, clipped_time, initial = 0)
        FVC = round(np.max(volume),2)
        
        # Calculate PEF, FVC, FVC1
        mx_peak = maxima(smoothen_envelope)
        peak_1s = smoothen_envelope[SAMPLE_RATE]
        
        # Convert the audio data into a suitable format for plotting
        envelope_data_points = [LineChartDataPoint(t, round(v,5)) for t, v in zip(sample_indices, smoothen_envelope)]
        
        # Converting the recorded audio signal into data points for plotting
        signal_data = [
            LineChartData(
                data_points=envelope_data_points,
                stroke_width=2,
                color=colors.BLUE_100,
                curved=True,
                stroke_cap_round=True,
                below_line_gradient=LinearGradient(
                    begin=alignment.top_center,
                    end=alignment.bottom_center,
                    colors=[
                        colors.with_opacity(0.5, colors.BLUE_100),
                        "transparent",
                    ],
                ),
            ),
        ]

        # Prepare the chart for plotting
        chart = LineChart(
            data_series=signal_data,
            border=Border(
                bottom=BorderSide(4, colors.with_opacity(0.5, colors.ON_SURFACE)),
                left=BorderSide(4, colors.with_opacity(0.5, colors.ON_SURFACE)),
            ),
            left_axis=ChartAxis(
                labels=[
                    ChartAxisLabel(
                        value=min(smoothen_envelope),
                        label=Text(f"{min(smoothen_envelope):.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                    ChartAxisLabel(
                        value=(min(smoothen_envelope) + max(smoothen_envelope)) / 2,
                        label=Text(f"{(min(smoothen_envelope) + max(smoothen_envelope)) / 2:.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                    ChartAxisLabel(
                        value=max(smoothen_envelope),
                        label=Text(f"{max(smoothen_envelope):.2f}", size=14, weight=FontWeight.BOLD),
                    ),
                ],
                labels_size=40,
                title=Text("Amplitude", size=13, weight=FontWeight.BOLD),
                
            ),
            bottom_axis=ChartAxis(
                labels=[
                    ChartAxisLabel(
                        value=sample_rate * 1,
                        label=Container(
                            content=Text(
                                "1s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                    ChartAxisLabel(
                        value=sample_rate * 2,
                        label=Container(
                            content=Text(
                                "2s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                    ChartAxisLabel(
                        value=sample_rate * 3,
                        label=Container(
                            content=Text(
                                "3s",
                                size=16,
                                weight=FontWeight.BOLD,
                                color=colors.with_opacity(0.5, colors.ON_SURFACE),
                            ),
                            margin=margin.only(top=10),
                        ),
                    ),
                ],
                labels_size=32,
                title=Text("Time(s)", size=13, weight=FontWeight.BOLD),
            ),
            tooltip_bgcolor=colors.with_opacity(0.8, colors.BLUE_GREY),
            min_y=min(smoothen_envelope),
            max_y=max(smoothen_envelope),
            min_x=0,
            max_x=DURATION * sample_rate,
            expand=False,
        )

        # Update the UI after processing
        page.controls.clear()
        page.add(
            Column(
                controls=[
                    Container(
                        content=Text("SpiroMask.ai", size=30, weight="bold", text_align=TextAlign.CENTER),
                        padding=padding.only(top=20),
                        alignment=alignment.center,
                    ),
                    Container(
                        content=chart,
                        padding=padding.only(top=20),
                        alignment=alignment.center,
                    ),
                    Row(
                        alignment=MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            Column(
                                controls=[
                                    Row(
                                        alignment=MainAxisAlignment.SPACE_BETWEEN,
                                        controls=[
                                            Container(
                                                content=Text(f"Max peaks: \n {mx_peak:.2f}", size=18, weight=FontWeight.BOLD, color=colors.WHITE),
                                                padding=10,
                                                bgcolor=colors.BLACK12,
                                                border_radius=5,
                                                margin=margin.only(left=10),
                                            ),
                                            Container(
                                                content=Text(f"Peak at 1s: \n {peak_1s:.2f}", size=18, weight=FontWeight.BOLD, color=colors.WHITE),
                                                padding=10,
                                                bgcolor=colors.BLACK12,
                                                border_radius=5,
                                                margin=margin.only(left=20, right=10),
                                                alignment=alignment.center_right,
                                            ),
                                        ],
                                    ),
                                    Row(
                                        controls=[
                                            Container(
                                                content=Text(f"Force Vital\n Capacity: {FVC}", size=17, weight=FontWeight.BOLD, color=colors.WHITE),
                                                padding=10,
                                                bgcolor=colors.BLACK12,
                                                border_radius=5,
                                                margin=margin.only(left=10),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Container(
                    #     content=IconButton(
                    #         icon=icons.RESTART_ALT,  # Use icon instead of text
                    #         on_click=lambda e: reset_page(),
                    #         icon_size=30,
                    #         tooltip="Record Again",
                    #     ),
                    #     alignment=alignment.center_right,
                    #     margin = margin.only(top=20),
                    #     padding=padding.only(right=10,bottom=250),
                    # ),
                ],
                alignment=alignment.center,
                expand=True,
            ),
            nav_bar()  # Add the navigation bar at the bottom
        )
        start_button.visible = True
        page.update()

    def reset_page():
        # Clear existing controls
        page.controls.clear()
        # status_text.page.padding = padding.only(top=10)
        status_text.value = 'SpiroMask.ai'
        instructions_text.visible = True
        # Add the initial state layout
        page.add(
            Column(
                controls=[
                    Container(
                        content=status_text,
                        padding=padding.only(top=50),  # Space from the top of the screen
                        alignment=alignment.center,
                    ),
                    Container(
                            content=instructions_text,
                            padding=padding.symmetric(horizontal=20, vertical=30),
                            alignment=alignment.center_left,
                        ),
                    Container(
                        content=start_button,
                        alignment=alignment.center,
                        padding=padding.only(top=20),  # Space between title and start button
                    ),
                ],
                alignment=alignment.center,
                expand=True,
            ),
            nav_bar()  # Add the navigation bar at the bottom
        )
        page.update()


    # Handler for route changes
    def route_change(route):
        page.controls.clear()
        if page.route == "/":
            # Default home page with image, title, and description
            page.add(
                Column(
                    controls=[
                        Container(
                            content=Image(src="home.png", width=100, height=100, border_radius=10),  # Adjust the path and size as needed
                            alignment=alignment.center,
                            padding=padding.only(top=200),
                        ),
                        Container(
                            content=Text("SpiroMask.ai", size=30, weight="bold", text_align=TextAlign.CENTER),
                            alignment=alignment.center,
                            padding=padding.only(top=10)
                        ),
                        Container(
                            content=Text(
                                "Your personal respiratory assistant, keeping track of your lung health anytime, anywhere.",
                                size=16,
                                text_align=TextAlign.CENTER,
                            ),
                            alignment=alignment.center,
                            padding=padding.symmetric(horizontal=20, vertical=10)
                        ),
                    ],
                    alignment=alignment.center,
                    expand=True,
                ),
                nav_bar()  # Add the navigation bar at the bottom
            )
        elif page.route == "/record":
            # Recording and processing page
            instructions_text.visible = True
            status_text.value = "SpiroMask.ai"
            
            # Instruction text
            instructions_text

            page.add(
                Column(
                    controls=[
                        Container(
                            content=status_text,
                            alignment=alignment.center,
                            padding=padding.only(top=30)
                        ),
                        Container(
                            content=instructions_text,
                            padding=padding.symmetric(horizontal=20, vertical=30),
                            alignment=alignment.center_left,
                        ),
                        Container(
                            content=start_button,
                            alignment=alignment.center,
                            padding=padding.only(top=30)
                        ),
                    ],
                    alignment=alignment.center,
                    expand=True,
                ),
                nav_bar()  # Add the navigation bar at the bottom
            ),

        page.update()

    page.on_route_change = route_change
    page.go("/")

app(target=main)
