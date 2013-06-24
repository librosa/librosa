# Test script for chromagram extraction. Set paths and exrtact chroma. Also time

import os
os.chdir("/Users/mattmcvicar/Desktop/Work/HPA_python_1.0")

import sys
sys.path.append("/Users/mattmcvicar/Desktop/Work/HPA_python_1.0/FeatureExtraction") 

# Name of wavefile to analyse
filename = '/Users/mattmcvicar/Desktop/Work/Python_stuff/chroma_extraction/pianoa4.wav';

import time

import HPA_extract_chroma as chromagram
reload(chromagram)
t1 = time.time()
[chroma, normal_chroma, sample_times, tuning] = chromagram.extract_chroma(filename)
t2 = time.time()

print 'chromagram extraction finished in %0.3f' % ((t2-t1))
print normal_chroma
