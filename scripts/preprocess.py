import os
import json
import argparse

from os import walk



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='LibriSpeech', help='Path to librispeech dataset')
    namespace = parser.parse_args()
    argv = vars(namespace)
    
    mypath = argv['path']
    waves = {}
    for (dirpath, dirnames, filenames) in walk(mypath):
        for name in filenames:
            if 'wav' in name:
                waves[name] = dirpath
