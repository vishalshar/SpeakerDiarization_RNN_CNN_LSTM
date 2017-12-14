import scipy.io.wavfile
import numpy as np

def get_data(filename, chunk_time=0.1, down_sample=False,down_sample_rate=4):

    # Get sample rate and read data from both channel
    sampleRate, data = scipy.io.wavfile.read(filename)
    num_of_frame_per_chunk=int(sampleRate*chunk_time)

    if num_of_frame_per_chunk!=sampleRate*chunk_time:
        raise ValueError('inappropriate chunk_time')

    # Two channels
    channel_1=data[:,0]
    channel_2=data[:,1]

    #abandon residue
    data_length = int(len(channel_1) // num_of_frame_per_chunk * num_of_frame_per_chunk)
    channel_1 = channel_1[:data_length]
    channel_2 = channel_2[:data_length]

    #reshape the data to a vector where each row is the data of a chunk
    data_matrix_1 = channel_1.reshape(-1, num_of_frame_per_chunk)
    data_matrix_2 = channel_2.reshape(-1, num_of_frame_per_chunk)

    if down_sample:
        data_matrix_1 = data_matrix_1[:, ::down_sample_rate]
        data_matrix_2 = data_matrix_2[:, ::down_sample_rate]
    #return a tuple of two matrixes, one for each channel
    return (data_matrix_1,data_matrix_2)



