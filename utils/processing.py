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
               
    
audio_transforms = nn.Sequential(
    torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'win_length':400, 'hop_length':160, "center":True, 'n_mels':64}),
    torchaudio.transforms.SlidingWindowCmn(cmn_window=600, norm_vars=True, center=True)
)

# Аугментации
train_audio_transforms = nn.Sequential(
#     torchaudio.transforms.SlidingWindowCmn(cmn_window=600, norm_vars=True, center=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
#     torchaudio.transforms.TimeMasking(time_mask_param=3)
)


def data_processing(data):
    features = []
    labels = []
    
    for waveform, target in data:
        spec = train_audio_transforms(waveform)
        features.append(spec)
        labels.append(target)
        
    features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return features, labels


def getTrainSamples(samples_number, train_samples, waves, nonspeech_dict):
    
    train_X = []
    train_Y = []
    for key in tqdm(list(train_samples.keys())[:samples_number]):
        path = waves[key] + '/' + key
        signal, samplerate = sf.read(path)

        target = np.array(train_samples[key])
        train_Y.append(target)
        
        # Добавление шумов (реальных и искусственных) в аудиозапись с заданым snr
        a = random.random()
        if a < 0.2:
            # white noise
            noise = get_white_noise(signal, SNR=3)
            signal_noise = signal + noise
        else:
            # real world noise
            b = random.randint(1,100)
            nonspeech_key = 'enc-n' + str(b) + '.wav'
            # Убираем начала и конец шума для более плавной интеграции
            nonspeech = nonspeech_dict[nonspeech_key][10000:-2000]
            if len(signal) > len(nonspeech):
                noise = np.pad(nonspeech, (0, len(signal) - len(nonspeech)), 'reflect', reflect_type='even')[:len(signal)]
            else:
                noise = nonspeech[:len(signal)]
            snr = random.randint(-3, 3)
            noise = get_noise_from_sound(signal, noise, SNR=snr)
            signal_noise = signal + noise
        mfcc = audio_transforms(torch.from_numpy(signal_noise).float())[:,:-1].transpose(0, 1)
        train_X.append(mfcc)
    
    return train_X, train_Y

def getTestSamples(max_samples, test_path, dev_samples):
    
    test_y = []
    test_X = []
    for key in tqdm(listdir(test_path)[:max_samples]):
        path = os.path.join(test_path, key)
        target = np.array(dev_samples[key])
        signal, samplerate = sf.read(path)
        
        
        mfcc = audio_transforms(torch.from_numpy(signal).float())[:,:-1].transpose(0, 1)
        test_X.append(mfcc)
        test_y.append(target)
    
    return train_X, train_y