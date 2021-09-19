#!/bin/bash
set -e

while getopts "f:c:" flag;
do
    case ${flag} in
        f) folder=${OPTARG};;
        *) echo 'error';;
    esac
done

# folder=Nonspeech/
# folder=/mnt/DATA2/shevchenko/vad/Nonspeech

for file in $(find "$folder" -type f -iname "*.wav")
do
    name=$(basename "$file" .wav)
    dir=$(dirname "$file")
    ffmpeg -y -i "$file" -ar 16000 "$dir"/enc-"$name".wav

done

# ffmpeg -i input.wav -ar 44100 output.wav