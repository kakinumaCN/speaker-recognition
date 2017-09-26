#_*_coding:utf-8_*_
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import wave
import numpy as np
from python_speech_features import mfcc
from sklearn.mixture import GMM
from sklearn import preprocessing

def get_MFCC(sr, audio):
    return preprocessing.scale(mfcc(audio, sr,  appendEnergy=False))

def train_GMM_N(data_path, model_path, dtype):
    if os.path.exists(os.path.join(data_path,'.DS_Store')):
        os.remove(os.path.join(data_path,'.DS_Store')) # for macOS
    speaker_list = os.listdir(data_path)
    # exit()
    N = len(speaker_list)
    # print (N)
    # exit()
    train_path = [0 for i in range(N)]
    # train_file_list = [([] * 3) for i in range(N)]
    train_mfcc_features = [0 for i in range(N)]
    speaker_gmm = [0 for i in range(N)]
    
    for i in range(N):
        # print i
        # if i < 61:
            # continue

        # train_path[i] = os.path.join(data_path,speaker_list[i])
        # train_file_list[i].append(speaker_list[i])
        # print (train_file_list[i])
        train_mfcc_features[i] = np.asarray((), dtype=dtype)

        f = wave.open(os.path.join(data_path,speaker_list[i]), 'rb')
        print os.path.join(data_path,speaker_list[i])
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
    #    print audio
        feature = get_MFCC(frame_rate, audio)
        if train_mfcc_features[i].size == 0:
            train_mfcc_features[i] = feature
        else:
            train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))

        speaker_gmm[i] = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))    

def test_GMM_N(data_path, model_path, dtype):
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    result = []
    for file in file_list:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        print (gmm_files[np.argmax(log_likelihood)])
        # print log_likelihood
        result.append(gmm_files[np.argmax(log_likelihood)])
    acc = 0
    for i in range(len(file_list)):
        print (file_list[i],result[i])
        if file_list[i].split('/')[2].split('.')[0] == result[i].split('/')[2].split('.')[0]:
            acc += 1
    print str(acc) + '/67'


if __name__ == '__main__':
    train_GMM_N('./train_data', './models', np.int16)
    # test_GMM_N('./test_data', './models', np.int16)
