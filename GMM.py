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
    speaker_list = ['1','2','3','4','5']
    N = len(speaker_list)
    train_path = [0 for i in range(N)]
    # train_path = np.zeros([0:N],dtype = np.int16)
    train_file_list = [([] * 3) for i in range(N)]
    train_mfcc_features = [0 for i in range(N)]
    speaker_gmm = [0 for i in range(N)]
    
    for i in range(N):
        # train_path.append(os.path.join(data_path,speaker_list[i]))
        print i
        train_path[i] = os.path.join(data_path,speaker_list[i])
        train_file_list[i] = [os.path.join(train_path[i], f) for f in os.listdir(train_path[i]) if f.endswith('.wav')]

        train_mfcc_features[i] = np.asarray((), dtype=dtype)

        for file in train_file_list[i]:
            f = wave.open(file, 'rb')
            frame_rate, n_frames = f.getframerate(), f.getnframes()
            audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
    #        print audio
            feature = get_MFCC(frame_rate, audio)
            if train_mfcc_features[i].size == 0:
                train_mfcc_features[i] = feature
            else:
                train_mfcc_features[i] = np.vstack((mfcc_features, feature))

        speaker_gmm[i] = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))    

    print train_path
    print train_file_list
    print train_mfcc_features

def test_GMM_N(data_path, model_path, dtype):
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    for file in file_list:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        print gmm_files[np.argmax(log_likelihood)]

if __name__ == '__main__':
    # train_GMM_N('./data', './models', np.int16)
    test_GMM_N('./test_data', './models', np.int16)
