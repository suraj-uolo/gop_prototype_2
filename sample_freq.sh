#!/bin/bash

for wav in $(awk '{print $2}' data/test/wav.scp); do
    sox "$wav" -r 16000 -b 16 -c 1 "${wav}_converted.wav"
done