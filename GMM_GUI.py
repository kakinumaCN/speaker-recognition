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
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

# import tkinter.filedialog as tf


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_MFCC(sr, audio):
    return preprocessing.scale(mfcc(audio, sr,  appendEnergy=False))

def train_GMM_N(speaker_list, model_path, dtype):
    # if os.path.exists(os.path.join(data_path,'.DS_Store')):
    #     os.remove(os.path.join(data_path,'.DS_Store')) # for macOS
    # speaker_list = os.listdir(data_path)
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

        f = wave.open(speaker_list[i], 'rb')
        # print os.path.join(data_path,speaker_list[i])
        print i
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
    #    print audio
        feature = get_MFCC(frame_rate, audio)
        if train_mfcc_features[i].size == 0:
            train_mfcc_features[i] = feature
        else:
            train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))

        # speaker_gmm[i] = GMM(n_components=4, n_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=3, max_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        print speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]+'.gmm'), 'w'))    

def test_GMM_N(file_list, model_path, dtype):
    gmm_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.gmm')]
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    # file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    print file_list
    result = []
    acc = 0
    for file in file_list:
        # print file
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        # print (gmm_files[np.argmax(log_likelihood)])
        # print log_likelihood
        if file.split('/')[len(file.split('/'))-1].split('.')[0] == gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0]:
            acc += 1
            print acc
        else:
            print gmm_files[np.argmax(log_likelihood)]
    # acc = 0
    # for i in range(len(file_list)):
    #     # print (file_list[i],result[i])
    #     if file_list[i].split('/')[2].split('.')[0] == result[i].split('/')[2].split('.')[0]:
    #         acc += 1
    print str(acc)

train_list = []
test_list = []

if __name__ == '__main__':
    import Tkinter as tk
    import tkFileDialog
    root= tk.Tk()
    root.title("说话人识别")
    root.geometry('500x200')
    path1 = tk.StringVar()
    path2 = tk.StringVar()


    # 读文件
    def selectPath1():
        # # path_ = tkFileDialog.askdirectory()
        # # path1.set(path_)
        # path = tkFileDialog.askopenfilenames()
        # # print path
        # # path_train_male.set(path)
        # train_path = path
        # global train_list
        # train_path = train_path[1:-1]
        # train_list = train_path.split(', ')
        # print train_list
        path = tkFileDialog.askopenfilenames()
        path1.set(path)
        train_path = path1.get().replace('\'','')
        global train_list
        train_path = train_path[1:-1]
        train_list = train_path.split(', ')

        print train_list[0]

    def selectPath2():
        path = tkFileDialog.askopenfilenames()
        path2.set(path)
        test_path = path2.get().replace('\'','')
        global test_list
        test_path = test_path[1:-1]
        test_list = test_path.split(', ')
    # 训练
    def train():
        # print 'train'
        # var1=entry1.get()
        train_GMM_N(train_list, 'b/models', np.int16)

    # 测试
    def test():
        # print 'test'
        # print path2.get()
        test_GMM_N(test_list, 'b/models', np.int16)
        # test_GMM_N('a/test_data', 'a/models', np.int16)

        # var2 = entry2.get()

    # 窗体定义
    tk.Label(root, text=' ').grid(row = 0, column = 0)
    tk.Label(root, text='说话人识别').place(x=165,y=20)

    tk.Label(root,text = "  ").grid(row = 1, column = 2)

    tk.Label(root,text = "      ").grid(row = 2, column = 0)
    tk.Label(root,text = "训练文件:").grid(row = 2, column = 1)
    entry1=tk.Entry(root, textvariable =path1,borderwidth = 3).grid(row = 2, column = 2)
    tk.Label(root,text = "  ").grid(row = 2, column = 3)
    tk.Button(root, text = "选择", command = selectPath1).grid(row = 2, column = 4)
    tk.Label(root,text = "    ").grid(row = 2, column = 5)
    tk.Button(root, text = "训练", command = train).grid(row = 2, column = 6)

    tk.Label(root,text = "  ").grid(row = 3, column = 2)

    tk.Label(root,text = "  ").grid(row = 4, column = 0)
    tk.Label(root,text = "测试文件:").grid(row = 4, column = 1)
    entry2=tk.Entry(root, textvariable =path2,borderwidth = 3).grid(row = 4, column = 2)
    tk.Label(root,text = "  ").grid(row = 4, column = 3)
    tk.Button(root, text = "选择", command = selectPath2).grid(row = 4, column = 4)
    tk.Label(root,text = "    ").grid(row = 4, column = 5)
    tk.Button(root, text = "测试", command = test).grid(row = 4, column = 6)

    tk.Label(root,text = "  ").grid(row = 5, column = 2)

    # tk.Label(root,text = "  测试正确率:").place(x=5,y=152)
    # tk.Entry(root, textvariable =accc,borderwidth = 3,width = 5,).place(x=85,y=152)
    # tk.Label(root,text = "  真实结果:").place(x=130,y=152)
    # tk.Entry(root, textvariable =' ',borderwidth = 3,width = 5,).place(x=200,y=152)
    # tk.Label(root,text = "  测试结果:").place(x=245,y=152)
    # tk.Entry(root, textvariable =' ',borderwidth = 3,width = 5,).place(x=315,y=152)

    root.mainloop()

    # train_GMM_N('a/train_data', 'a/models', np.int16)

    # n_component
    #  256 - 161
    #  128 - 176
    #  64 - 184 
    #  48 - 187

    #  2 - 153
    #  3 - 166
    #  4 - 175
    #  8 - 182
    # 12 - 185
    # 24 - 189

    # 24
    # 50 - 46
    # 100 - 92
    # 150 - 140


    # vad
    # 24 - 177
    # 8 - 162
    # 48 - 180
