#_*_coding:utf-8_*_
import os
import sys
import getopt
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import wave
import numpy as np
from python_speech_features import mfcc
# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

def get_MFCC(sr, audio):
    return preprocessing.scale(mfcc(audio, sr,  appendEnergy=False))

def train_GMM(data_path, model_path, dtype):
    # - train_data
    #   ├ 1.wav
    #   ├ 2.wav
    #   └ ...
    speaker_list = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.wav')] # without'.wav'
    # speaker_list = [f for f in os.listdir(data_path) if f.endswith('.wav')] # with '.wav'
    print speaker_list
    N = len(speaker_list)
    train_mfcc_features = [0 for i in range(N)]
    speaker_gmm = [0 for i in range(N)]

    for i in range(N):
        f = wave.open(os.path.join(data_path,speaker_list[i] + '.wav'), 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)

        feature = get_MFCC(frame_rate, audio)
        print n_frames, feature.shape
        train_mfcc_features[i] = feature

        # if train_mfcc_features[i].size == 0:
        #     train_mfcc_features[i] = feature
        # else:
        #     train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))

        # speaker_gmm[i] = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))   

def test_GMM(data_path, model_path, dtype):
    # - test_data
    #   ├ 1.wav
    #   ├ 2.wav
    #   └ ...
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    result = []
    count = 0
    for file in file_list:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        print file, gmm_files[np.argmax(log_likelihood)]
        # count
        if file.split('/')[2].split('.')[0] ==  gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0] :
            count = count + 1
        print count

if __name__ == '__main__':
    
    opts, args = getopt.getopt(sys.argv[1:],'',['train','test'])
    for opt, arg in opts:
        if opt in ('--train'):
            train_GMM('../train_data', '../models', np.int16)
        elif opt in ('--test'):
            test_GMM('../test_data', '../models', np.int16)