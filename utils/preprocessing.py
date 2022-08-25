import numpy as np
from scipy import signal
from scipy.signal import iirnotch, butter, filtfilt


def notch_filtering(Frequency, SelectedVector, SamplingRate, Q=35):
    """remove single  Frequency from signal"""
    wo = Frequency / (SamplingRate / 2)
    # bw= wo / Q
    # Make order 2 bandstop digital filter
    b, a = iirnotch(wo, Q)
    notch_filtered = signal.lfilter(b, a, SelectedVector)
    return notch_filtered


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def preprocess_signal(
    original_signal,
    SamplingRate=500,
    LF=60,
    HF=240,
    frequences_to_filter=[50, 100, 150, 200],
    order_butter=4,
    save_plot=False,
):
    """detrend, apply notch filter to remove given frequences, apply butter filter and return the envelop of processed signal"""
    print("SHAPE", original_signal.shape)
    signal_emg_detrended = signal.detrend(original_signal)
    # apply notch
    notch_filtered = signal_emg_detrended.copy()
    for filtered_frequency in frequences_to_filter:
        notch_filtered = notch_filtering(
            filtered_frequency, notch_filtered, SamplingRate, Q=35
        )

    # apply band filter on filtered signal
    processed = butter_bandpass_filter(
        notch_filtered, LF, HF, SamplingRate, order_butter
    )
    # evelope
    processed_H = signal.hilbert(processed)
    amplitude_envelope = np.abs(processed_H)
    if save_plot:
        # final spectr after processing
        from matplotlib import pyplot as plt

        plt.figure(2)
        plt.clf()
        plt.magnitude_spectrum(
            original_signal, Fs=SamplingRate, label="Noisy signal magnitude"
        )
        plt.axis("tight")

        plt.magnitude_spectrum(
            processed, Fs=SamplingRate, label="Filtered final signal magnitude"
        )
        plt.grid(True)
        plt.axis("tight")
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig("figures/processed_vs_original.png")
    return amplitude_envelope
