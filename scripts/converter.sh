#!/bin/bash

folder=/mnt/DATA2/shevchenko/vad/LibriSpeech/train-clean-360

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    ffmpeg -i "$file" "$dir"/"$name".wav
    #ffmpeg -i $file $dir/$name.wav
    rm "$file"
done