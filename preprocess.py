import os
import json
import pickle
import random
import argparse
import webrtcvad
import subprocess

import numpy as np
import soundfile as sf

from os import listdir
from webrtc_utils import *
from tqdm import tqdm
from os import walk
from collections import defaultdict
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_libri', default='LibriSpeech/', help='Path to librispeech dataset')
    parser.add_argument('--path_nonspeech', default='Nonspeech/', help='Path to nonspeech data')
    namespace = parser.parse_args()
    argv = vars(namespace)
    

#     bash_command = "bash converter.sh -f {}".format(argv['path_libri'])
#     subprocess.run(bash_command.split(), shell=True, check=True)
    
#     bash_command = "bash changesr.sh -f {}".format(argv['path_nonspeech'])
#     subprocess.run(bash_command.split(), shell=True, check=True)
    
    mypath = argv['path_libri']
    waves = {}
    for (dirpath, dirnames, filenames) in walk(mypath):
        for name in filenames:
            if 'wav' in name:
                waves[name] = dirpath
    
    with open('waves.json', 'w') as f:
        json.dump(waves, f)
                
    vad_dict = defaultdict(list)
    vad = webrtcvad.Vad(1)
    for name, dirpath in tqdm(waves.items()):
        vads = []
        path = dirpath + '/' + name
        audio, sample_rate = read_wave(path)

        frames = frame_generator(10, audio, sample_rate)
        frames = list(frames)

        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            speech = 1 if is_speech else 0
            vads.append(speech)
        vad_dict[name] = vads
    
    
    dev_samples = {}
    train_samples = {}

    keys_array = np.array(list(vad_dict.keys()))

    train_keys, dev_keys = train_test_split(keys_array, test_size=0.2)

    for key in train_keys:
        train_samples[key] = vad_dict[key]
    for key in dev_keys:
        dev_samples[key] = vad_dict[key]
        
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_samples, f, pickle.HIGHEST_PROTOCOL)    

    with open('dev.pkl', 'wb') as f:
        pickle.dump(dev_samples, f, pickle.HIGHEST_PROTOCOL)   

    with open('all.pkl', 'wb') as f:
        pickle.dump(vad_dict, f, pickle.HIGHEST_PROTOCOL)  
    
    
    nonspeech_path = argv['path_nonspeech']
    dev_mfccs = []
    nonspeech_dict = {}
    for i in listdir(nonspeech_path):
        if 'enc' in i:
            path = os.path.join(nonspeech_path, i)
            signal, samplerate = sf.read(path)
            nonspeech_dict[i] = signal
    
    # Сохраняем тестовые данные заранее и не меняем им
    if not os.path.exists('test/'):
        os.mkdir('test/')
        
    for key in tqdm(dev_samples):
        path = waves[key] + '/' + key
        signal, samplerate = sf.read(path)
        a = random.random()
        # Добавление шумов (реальных и искусственных) в аудиозапись с заданым snr
        if a < 0.2:
            # white noise
            noise = get_white_noise(signal, SNR=3)
            signal_noise = signal + noise
            sf.write('test/' + key, signal_noise, samplerate)
        else:
            # real world noise
            b = random.randint(1,100)
            nonspeech_key = 'enc-n' + str(b) + '.wav'
            nonspeech = nonspeech_dict[nonspeech_key][10000:-2000]
            if len(signal) > len(nonspeech):
                noise = np.pad(nonspeech, (0, len(signal) - len(nonspeech)), 'reflect', reflect_type='even')[:len(signal)]
            else:
                noise = nonspeech[:len(signal)]
            noise = get_noise_from_sound(signal, noise, SNR=3)
            signal_noise = signal + noise
            sf.write('test/' + key, signal_noise, samplerate)

