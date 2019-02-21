# SpeakerDiarization_RNN
Speaker diarization problem using Recurrent Neural Network. 
Speaker Diarization is the problem of separating speakers in an audio. There could be any number of speakers and final result should state when speaker starts and ends. In this project, we analyze given audio file with 2 channels and 2 speakers (on separate channel).

# Data
Data used in the process cannot be shared because of privacy concerns but if you need to test this code I can provide one sample data to try this code and test. Please email me for the sample data.

## Dataset Description
Our dataset contains 37 audio files approximately of 15 minutes each with sampling rate of 44100 samples/second, recorded in 2 channels with exactly 2 speakers on 2 different microphones. Each audio file has been hand annotated for speakers timings. Annotating timing (in seconds) they start and stop speaking. We use this dataset and split in 3 parts for training, validation and testing.

## Preprocessing

### Data Normalization
We perform normalization of audio files after observing recorded audio was not in the same scale. Few audio files were louder than others and normalization can help bring all audio files to same scale.

### Sampling Audio
With frame rate being high, we have a lot of data. To give an example, in a 15 min audio file we get about 40M samples in each channel.  To reduce data without loosing much information, we down sample audio files by every 4 sample. 


## Multi-layer Perceptron (MLP)
Code for MLP is in file MLP_1201_2.py
## Recurrent Neural Network (RNN)
Code for RNN is in Alg4_RNN_1channel_2classes.py
## Convolution Neural Network (CNN)
Code for CNN is in CNN_1channel_2classes.py
## Documentation is in folder /documentation
Documentation: speaker-diarization-recurrent.pdf
