#!/bin/bash

folder=/mnt/DATA2/shevchenko/vad/Nonspeech


for file in $(find "$folder" -type f -iname "*.wav")
do
    name=$(basename "$file" .wav)
    dir=$(dirname "$file")
    ffmpeg -y -i "$file" -ar 16000 "$dir"/enc-"$name".wav
    #ffmpeg -i $file $dir/$name.wav
#     rm "$file"
done

# ffmpeg -i input.wav -ar 44100 output.wav