import math
import numpy as np

def get_white_noise(signal, SNR) :
    #RMS value of signal
    RMS_s = math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n = math.sqrt(RMS_s**2 / pow(10, SNR / 10))

    noise = np.random.normal(0, RMS_n, signal.shape[0])
    return noise

#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal, noise, SNR):
    RMS_s = math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n = math.sqrt(RMS_s**2 / pow(10, SNR / 10))
    
    #current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise**2))
    noise = noise * (RMS_n / RMS_n_current)
    
    return noise

# modified version from webrtc github
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n