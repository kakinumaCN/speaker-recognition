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
    speaker_list = [f for f in os.listdir(data_path) if not f.startswith('.')]  # remove .DS_Store
    n = len(speaker_list)
    train_mfcc_features = [np.zeros(0) for i in range(n)]
    speaker_gmm = [0 for i in range(n)]

    for i in range(n):
        file_list = [f for f in os.listdir(os.path.join(data_path, speaker_list[i])) if f.endswith('.wav')]
        file_list.sort(key = str.lower)
        fitcount = 400
        for file1 in file_list:
            f = wave.open(os.path.join(data_path, speaker_list[i], file1), 'rb')
            # print speaker_list[i],file
            frame_rate, n_frames = f.getframerate(), f.getnframes()
            audio = np.fromstring(f.readframes(n_frames), dtype=dtype)

            feature = get_mfcc(frame_rate, audio)
            if train_mfcc_features[i].size == 0:
                train_mfcc_features[i] = feature
            else:
                train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))

            # for share database
            print file1
            fitcount -= 1  # 300
            if fitcount <= 2:  # 300
                break  # 300

        # speaker_gmm[i] = GMM(n_components=8, n_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200, covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        print 'fit' + speaker_list[i]
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i]+'.gmm'), 'w'))


def test_gmm(data_path, model_path, dtype):
    # - test_data
    #   ├ 1.wav
    #   ├ 2.wav
    #   └ ...
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    for file in file_list:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_mfcc(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        print file, gmm_files[np.argmax(log_likelihood)]


def test_gmm_path(data_path, model_path, dtype):
    # - test_data
    #   ├ a
    #     ├ a001.wav
    #     └ ...
    #   ├ b
    #     ├ b001.wav
    #     └ ...
    #   └ ...
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    speaker_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if not f.startswith('.')]
    file_list = []
    for s in speaker_list:
        f = [os.path.join(s, file) for file in os.listdir(s) if file.endswith('.wav')]
        file_list.extend(f[402:])
    for file in file_list:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_mfcc(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        print file, gmm_files[np.argmax(log_likelihood)]


if __name__ == '__main__':

    DATAPATH = '../testpath'
    MODELPATH = '../testmodel'
    RUNTYPE = '2'

    opts, args = getopt.getopt(sys.argv[1:], 'hd:m:t:', ['help','datapath=','modelpath=','runtype=','train', 'test'])
    if len(opts) < 1:
        print 'run \'python gmm.py --help\' for help'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print 'usage: python gmm.py [-h|--help] [-d|--datapath <datapath>] [-m|--modelpath <modelpath>] [-t|--runtype 1|2] [--train] [--test]'
            print 'arguments: choose runtype 1 to run train()\n           choose runtype 2 to run train_path()'
            print 'default:'+' datapath:'+DATAPATH+' modelpath:'+MODELPATH+' runtype:'+RUNTYPE
            print 'examples: python gmm.py -d ../persondata/ -m ../models/ -t 2 --train --test'
        elif opt in ('-d', '--datapath'):
            DATAPATH = arg
        elif opt in ('-m', '--modelpath'):
            MODELPATH = arg
        elif opt in ('-t', '--runtype'):
            RUNTYPE = arg
        elif opt in '--train':
            if RUNTYPE == '2':
                train_gmm_path(DATAPATH, MODELPATH, np.int16)
            elif RUNTYPE == '1':
                train_gmm(DATAPATH, MODELPATH, np.int16)
        elif opt in '--test':
            if RUNTYPE == '2':
                PATH1 = [DATAPATH + p for p in os.listdir(DATAPATH)]
                for P in PATH1:
                    test_gmm_path(P, MODELPATH, np.int16)
            elif RUNTYPE == '1':
                test_gmm(DATAPATH, MODELPATH, np.int16)
            else:
                print '?'