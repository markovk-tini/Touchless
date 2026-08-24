#!/bin/bash

# Create a test AAC file
ffmpeg -f lavfi -i "anullsrc=r=48000:cl=stereo" -f lavfi -i "sine=f=440:r=48000:d=1" -filter_complex "[0][1]amix=inputs=2" -c:a aac -b:a 192k -t 1 /tmp/test_input.aac -y 2>&1 | tail -5

# Now test concat with aevalsrc + AAC
ffmpeg \
  -f lavfi -i "aevalsrc=exprs=0|0:c=stereo:s=48000:d=0.5" \
  -i /tmp/test_input.aac \
  -filter_complex "[0][1:a]concat=n=2:v=0:a=1" \
  -f null - 2>&1 | tail -30

