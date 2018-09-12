# coding:utf-8
import os
import sys
import getopt
import wave
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_mfcc(sr, audio):
    # return preprocessing.scale(mfcc(audio, sr,  appendEnergy=False))
    processed_audio = preprocessing.scale(mfcc(audio, sr, appendEnergy=False))
    delta1 = delta(processed_audio, 1)
    delta2 = delta(processed_audio, 2)
    ft = np.hstack((processed_audio, delta1, delta2))
    return ft


def train_gmm(data_path, model_path, dtype):
    # - train_data
    #   ├ 1.wav
    #   ├ 2.wav
    #   └ ...
    speaker_list = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.wav')]  # without'.wav'
    # speaker_list = [f for f in os.listdir(data_path) if f.endswith('.wav')] # with '.wav'
    print speaker_list
    n = len(speaker_list)
    train_mfcc_features = [0 for i in range(n)]
    speaker_gmm = [0 for i in range(n)]

    for i in range(n):
        print 'fit'+speaker_list[i]
        f = wave.open(os.path.join(data_path, speaker_list[i] + '.wav'), 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)

        feature = get_mfcc(frame_rate, audio)
        train_mfcc_features[i] = feature

        speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))


def train_gmm_path(data_path, model_path, dtype):
    # - train_data
    #   ├ a
    #     ├ 1.wav
    #     └ ...
    #   ├ b
    #     ├ 1.wav
    #     └ ...
    #   └ ...
    # speaker_list = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.wav')] # without'.wav'
    # speaker_list = [f for f in os.listdir(data_path) if f.endswith('.wav')] # with '.wav'
    speaker_list = [f for f in os.listdir(data_path) if not f.startswith('.')]  # remove .DS_Store
    n = len(speaker_list)
    train_mfcc_features = [np.zeros(0) for i in range(n)]
    speaker_gmm = [0 for i in range(n)]

    for i in range(n):
        file_list = [f for f in os.listdir(os.path.join(data_path, speaker_list[i])) if f.endswith('.wav')]
        fitcount = 400  # 300
        for file1 in file_list:
            f = wave.open(os.path.join(data_path, speaker_list[i], file1), 'rb')
            # print speaker_list[i],file
            frame_rate, n_frames = f.getframerate(), f.getnframes()
            audio = np.fromstring(f.readframes(n_frames), dtype=dtype)

            feature = get_mfcc(frame_rate, audio)
            if train_mfcc_features[i].size == 0:
                train_mfcc_features[i] = feature
                # print 'new feature' ,train_mfcc_features[i].shape
            else:
                train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))
                # print 'add feature' ,feature.shape, train_mfcc_features[i].shape

            fitcount -= 1  # 300
            if fitcount < 1:  # 300
                break  # 300

        # speaker_gmm[i] = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        print 'fit' + speaker_list[i]
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))


# def test_gmm(data_path, model_path, dtype):
#     # - test_data
#     #   ├ 1.wav
#     #   ├ 2.wav
#     #   └ ...
#     gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
#     models = [pickle.load(open(f, 'r')) for f in gmm_files]
#
#     file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
#     result = []
#     count = 0
#     for file in file_list:
#         f = wave.open(file, 'rb')
#         frame_rate, n_frames = f.getframerate(), f.getnframes()
#         audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
#         feature = get_mfcc(frame_rate, audio)
#         log_likelihood = np.zeros(len(models))
#         for i in range(len(models)):
#             scores = np.array(models[i].score(feature))
#             log_likelihood[i] = scores.sum()
#         print file, gmm_files[np.argmax(log_likelihood)]
#         # count
#         if file.split('/')[2].split('.')[0] ==  gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0] :
#             count = count + 1
#         print count
#
#
# def test_gmm_path(data_path, model_path, dtype):
#     # - test_data
#     #   ├ a
#     #     ├ 400.wav
#     #     └ ...
#     #   ├ b
#     #     ├ 400.wav
#     #     └ ...
#     #   └ ...
#     gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
#     models = [pickle.load(open(f, 'r')) for f in gmm_files]
#     speaker_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if not f.startswith('.')]
#     file_list = []
#     for s in speaker_list:
#         f = [os.path.join(s, file) for file in os.listdir(s) if file.endswith('.wav')]
#         file_list.extend(f[305:])
#     result = []
#     count = 0
#     xcount = 0
#     for file in file_list:
#         f = wave.open(file, 'rb')
#         frame_rate, n_frames = f.getframerate(), f.getnframes()
#         audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
#         feature = get_mfcc(frame_rate, audio)
#         log_likelihood = np.zeros(len(models))
#         for i in range(len(models)):
#             scores = np.array(models[i].score(feature))
#             log_likelihood[i] = scores.sum()
#         print file, gmm_files[np.argmax(log_likelihood)]
#         # count
#         # if file.split('/')[2].split('.')[0] ==  gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0] :
#         #     count = count + 1
#         # print count
#         # cout /Volumes/Storage/IOS/data/wav/C1_110/C0001/IC0001W0480.wav ../models/C0001.gmm
#         if file.split('/')[7] ==  gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0] :
#             count = count + 1
#         xcount += 1
#         print str(count)+'/'+str(xcount)


if __name__ == '__main__':

    # DATAPATH = '/Volumes/Storage/IOS/data/wav/'
    DATAPATH = '../testpath'
    MODELPATH = '../models_39mfcc_64'

    opts, args = getopt.getopt(sys.argv[1:], '', ['train', 'test'])

    # PATH1 = [DATAPATH+'D1_100', DATAPATH+'1']
    # PATH1 = [DATAPATH + p for p in os.listdir(PATH1) if p.startswith('D') and not p.startswith('D1_') \
    # and not p.startswith('D101_')]

    for opt, arg in opts:
        if opt in '--train':
            # for P in PATH1:
            #     train_gmm_path(P, MODELPATH, np.int16)
            train_gmm(DATAPATH, MODELPATH, np.int16)
        elif opt in '--test':
            for P in PATH1:
                pass
