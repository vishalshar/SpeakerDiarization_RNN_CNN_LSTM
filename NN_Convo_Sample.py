import scipy.io.wavfile
import os

# directory = "/media/vishal/Share/Diarization/Project Data/Sound Files/"
directory = "./Convo_Sample.wav"

# Get sample rate and read data from both channel
sampleRate, data = scipy.io.wavfile.read(directory)

# Two channels
channel_1=data[:,0]
channel_2=data[:,1]

# Convert data to mono channel
monoChannel = data.mean(axis=1)

# Print to test
print sampleRate
print len(data)
# print data[0:100]
print type(data)


