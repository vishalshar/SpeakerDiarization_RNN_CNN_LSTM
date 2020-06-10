## Citation
If you find our project helpful please cite our arxiv paper below:

```

```

# SpeakerDiarization
Speaker Diarization is the problem of separating speakers in an audio. There could be any number of speakers and final result should state when speaker starts and ends. In this project, we analyze given audio file with 2 channels and 2 speakers (on separate channel). We train Neural Network for learning when a person is speaking. We use different type of Neural Networks specifically, Single Layer Perceptron (SLP), Multi Layer Perceptron (MLP), Recurrent Neural Network (RNN) and Convolution Neural Network (CNN) we achieve 92% of accuracy with RNN. 


# Data
Data used in the process cannot be shared because of privacy concerns but if you need to test this code I can provide one sample data to try this code and test. Please email me for the sample data.

## Dataset Description
Our dataset contains 37 audio files approximately of 15 minutes each with sampling rate of 44100 samples/second, recorded in 2 channels with exactly 2 speakers on 2 different microphones. Each audio file has been hand annotated for speakers timings. Annotating timing (in seconds) they start and stop speaking. We use this dataset and split in 3 parts for training, validation and testing.


## Preprocessing

### Data Normalization
We perform normalization of audio files after observing recorded audio was not in the same scale. Few audio files were louder than others and normalization can help bring all audio files to same scale.

### Sampling Audio
With frame rate being high, we have a lot of data. To give an example, in a 15 min audio file we get about 40M samples in each channel.  To reduce data without loosing much information, we down sample audio files by every 4 sample. 



### Cleaning Labels
Provided labels needed some cleaning described below:
* Names of the speakers was not consistent throughout the data file, we cleaned it and made sure name is consistent.
* File also contained unicode, which needed to be cleaned. Python goes crazy with unicodes lol
* There were miss alignments as well in the data and needed to be removed and fixed.


# Approach 

## Multi-layer Perceptron (MLP)
We start with a basic single layer perceptron model. We implement 3 different models with hidden layer of different sizes 100, 200, 500 neurons. We achieve approximately 86\% accuracy. 
We next move to multi-layer perceptron model and try models with 2 layers deep. First layer had 100 and second 50 neurons and another with higher number of neurons (First Layer: 200, Second Layer: 100) (First Layer: 300, Second Layer: 50). For all the networks used in this project, the hidden neurons are ReLu \cite{relu} and the output neuron are sigmoid. The cost function used is cross entropy and mini-batch gradient descent with Adam optimization is used to train network.

Code for MLP is in file MLP_1201_2.py

## Recurrent Neural Network (RNN)
Next we try Recurrent Neural Network on the classification problem. The RNN gives us the best result with 3 layers each with 150 Long short-term memory (LSTM) cells. The LSTM in the graph means a LSTM layer which consists of 150 LSTM cells. The output only has one neuron with sigmoid to predict 0 or 1. 

![LSTM Network Architecture](https://github.com/vishalshar/SpeakerDiarization_RNN_CNN_LSTM/blob/master/documentation/speaker-diarization-recurrent/RNN.png)

## Convolution Neural Network (CNN)
To apply CNN, we at first compute the spectrogram for each row of the data matrix, then store them into a new file by using pickle. In this way we don’t need to compute spectrogram online and hence can save a lot of training time. Function scipy. signal.spectrogram is used to compute the spectrogram for each segment. The recomputed spectrogram of each segment then is organized to a 3 dimension matrix with shape (number of segments, height, width). For example, the down sampled data matrix of a channel returned by get data has the shape (100, 1102) for a channel with 100 segments, then the shape of recomputed spectrogram matrix is (100,129,4). The number of segments remains the same. The height 129 and width 4 come from using the default parameters of function scipy. signal.spectrogram. Spectrogram matrices are computed and stored by using code in Spectrogram Generator.

![CNN Network Architecture](https://github.com/vishalshar/SpeakerDiarization_RNN_CNN_LSTM/blob/master/documentation/speaker-diarization-recurrent/CNN.png)


# Results

![MLP](https://github.com/vishalshar/SpeakerDiarization_RNN_CNN_LSTM/blob/master/documentation/speaker-diarization-recurrent/MLP_6.png)
![CNN](https://github.com/vishalshar/SpeakerDiarization_RNN_CNN_LSTM/blob/master/documentation/speaker-diarization-recurrent/CNN_1.png)
![RNN](https://github.com/vishalshar/SpeakerDiarization_RNN_CNN_LSTM/blob/master/documentation/speaker-diarization-recurrent/RNN_4.png)
