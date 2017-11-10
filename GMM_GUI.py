#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import wave
import numpy as np
np.set_printoptions(threshold=np.inf)
from python_speech_features import mfcc
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

import codecs
import json
import math
# import tkinter.filedialog as tf
correctrate=float()

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
        print feature
        if train_mfcc_features[i].size == 0:
            train_mfcc_features[i] = feature
        else:
            train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))

        # speaker_gmm[i] = GMM(n_components=4, n_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=3, max_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        print speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]+'.gmm'), 'w'))    

def test_GMM_N_p(file_list, model_path, dtype):
    gmm_files = []
    for f in os.listdir(model_path):
        if f.endswith('.gmm'):
            gmm_files.append(model_path+'/'+f)
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    # file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    # print 'file_list'+file_list
    result = []
    acc = 0
    # reportJson = {}
    # resultList = []
    for file in file_list:
        # print file
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))

        if True:
            # 11.1
            # reportFile = codecs.open(file.split('/')[len(file.split('/'))-1].split('.')[0]+'_report.json', 'w', 'utf-8')
            # fileJson = {}
            # 10.30 20:00
            # fileJson['testFilename'] = file
            # fileJson['pwiList'] = []
            # loglike = []
            # for log in log_likelihood:
            #     loglike.append(log)
            # fileJson['likelihoodList'] = loglike
            

            for i in range(len(models)):
                #11.1
                # scores = np.array(models[i].score(feature))
                # log_likelihood[i] = scores.sum()

                # print models[i].score_samples(feature)
                print 'sum:'
                # print models[i]._estimate_weighted_log_prob(feature)
                # print 'log-prob:'
                # print models[i]._estimate_log_prob(feature)
                # print 'log-weight:'
                # print models[i]._estimate_log_weights()
                pwilist = []
                # for m in range(len(models[i]._estimate_weighted_log_prob(feature))):

                for m in range(0,500):
                    # print m
                    #each sample
                    pwi = 0
                    # print models[i]._estimate_weighted_log_prob(feature)[m]
                    for n in range(0,len(models[i]._estimate_weighted_log_prob(feature)[m])):
                        #each component
                        if models[i]._estimate_weighted_log_prob(feature)[m][n] > -9:
                             pwi += math.exp(models[i]._estimate_weighted_log_prob(feature)[m][n])
                        else :
                            pass
                    pwilist.append(pwi)
                print gmm_files[i]
                print pwilist[np.argmax(pwilist)]
                # fileJson['pwiList'].append(pwilist[np.argmax(pwilist)])
                
                    # print 'pwi is ' + str(pwi)
            # fileJson['bestPwiList'] = []
            # fileJson['bestMatchFileList'] = []

            # fileJson['bestPwiList'].append(fileJson['pwiList'][np.argmax(fileJson['pwiList'])])
            # fileJson['bestMatchFileList'].append(gmm_files[np.argmax(fileJson['pwiList'])])

            # fileJson['pwiList'][np.argmax(fileJson['pwiList'])] = -1
            # fileJson['bestPwiList'].append(fileJson['pwiList'][np.argmax(fileJson['pwiList'])])
            # fileJson['bestMatchFileList'].append(gmm_files[np.argmax(fileJson['pwiList'])])

            # fileJson.pop('pwiList')
            # json.dump(fileJson,reportFile)


        # fileJson['pwi'] = pwilist[np.argmax(pwilist)]
        # pwilist[np.argmax(pwilist)] = -1
        # fileJson['pwiFile'] = gmm_files[np.argmax(pwilist)]
        # fileJson['pwiSecond'] = pwilist[np.argmax(pwilist)]
        # fileJson['pwiSecondFile'] = pwilist[np.argmax(pwilist)]
            

            

            # a,b = models[i]._estimate_log_prob_resp(feature)
            # print a
            # print b
            
            # print models[i].predict_proba(feature)
        # print (gmm_files[np.argmax(log_likelihood)])
        # print log_likelihood
        # print file.split('/')[len(file.split('/'))-1].split('.')[0]
        # print gmm_files
        # print gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]
        # if file.split('/')[len(file.split('/'))-1].split('.')[0] == gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0]:
        if file.split('/')[len(file.split('/'))-1].split('.')[0] ==gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]:
    #if 当前文件名 == 测试结果文件名
            acc += 1
            # print acc
        else:
            pass
        # print log_likelihood
        # print log_likelihood[np.argmax(log_likelihood)]
        # print np.argmax(log_likelihood)
        # print gmm_files[np.argmax(log_likelihood)]
    # acc = 0
    # for i in range(len(file_list)):
    #     # print (file_list[i],result[i])
    #     if file_list[i].split('/')[2].split('.')[0] == result[i].split('/')[2].split('.')[0]:
    #         acc += 1
        

        # if file.split('/')[len(file.split('/'))-1].split('.')[0] ==gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]:
        #     fileJson['max'] = log_likelihood[np.argmax(log_likelihood)]

        # resultList.append(fileJson)
        # print fileJson
    # reportJson['resultList'] = resultList
    # json.dump(reportJson,reportFile)
    # print reportJson

    print '正确个数'+str(acc)
    print len(file_list)
    correctrate=float(acc)/float(len(file_list))*100
    print correctrate
    e.set(str(correctrate))
    # print '正确率'+int(acc)/len(file_list)  #正确个数/总个数
    if len(file_list) ==1:
        original_e.set(file.split('/')[len(file.split('/'))-1].split('.')[0])
        flag_e.set(gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0])
    else:
        original_temp=''
        flag_temp=''
        original_e.set(original_temp)
        flag_e.set(flag_temp)
    
def test_GMM_N(file_list, model_path, dtype):
    gmm_files = []
    for f in os.listdir(model_path):
        if f.endswith('.gmm'):
            gmm_files.append(model_path+'/'+f)
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    # file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')]
    # print 'file_list'+file_list
    result = []
    acc = 0
    reportFile = codecs.open('gmm_report.json', 'w', 'utf-8')
    reportJson = {}
    resultList = []
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
        # print file.split('/')[len(file.split('/'))-1].split('.')[0]
        # print gmm_files
        # print gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]
        # if file.split('/')[len(file.split('/'))-1].split('.')[0] == gmm_files[np.argmax(log_likelihood)].split('/')[2].split('.')[0]:
        if file.split('/')[len(file.split('/'))-1].split('.')[0] ==gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]:
        #if 当前文件名 == 测试结果文件名
            acc += 1
            # print acc
        else:
            pass
        # print log_likelihood
        # print log_likelihood[np.argmax(log_likelihood)]
        # print np.argmax(log_likelihood)
        # print gmm_files[np.argmax(log_likelihood)]
    # acc = 0
    # for i in range(len(file_list)):
    #     # print (file_list[i],result[i])
    #     if file_list[i].split('/')[2].split('.')[0] == result[i].split('/')[2].split('.')[0]:
    #         acc += 1
        fileJson = {}
        
        # 10.30 20:00
        fileJson['testFilename'] = file
        fileJson['modelList'] = gmm_files
        loglike = []
        for log in log_likelihood:
            loglike.append(log)
        fileJson['likelihoodList'] = loglike
        loglikeArray = np.array(loglike)
        minloglike = np.argsort(-loglikeArray)[0:3]
        
        print minloglike

        arraya = []
        arrayb = []
        arrayc = []
        for i in range(0,3):
            print minloglike[i]
            arraya.append(loglike[minloglike[i]])
            arrayb.append(gmm_files[minloglike[i]])
            arrayc.append(models[minloglike[i]])
        fileJson['likelihoodList'] = arraya
        fileJson['modelList'] = arrayb
        # fileJson['likelihoodList'] =
        # print nploglike[minloglike[0:5]]
        # fileJson['modelList'] = gmm_files[minloglike]
        arrayd = []
        for i in range(len(arrayc)):
                #11.1
                # scores = np.array(models[i].score(feature))
                # log_likelihood[i] = scores.sum()

                # print models[i].score_samples(feature)
                # print models[i]._estimate_weighted_log_prob(feature)
                # print 'log-prob:'
                # print models[i]._estimate_log_prob(feature)
                # print 'log-weight:'
                # print models[i]._estimate_log_weights()
            pwilist = []
            for m in range(500,1300): 
            # for m in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature))):
                # print m
                #each sample
                pwi = 0
                # print models[i]._estimate_weighted_log_prob(feature)[m]
                for n in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature)[m])):
                    #each component
                    if arrayc[i]._estimate_weighted_log_prob(feature)[m][n] > -9:
                            pwi += math.exp(arrayc[i]._estimate_weighted_log_prob(feature)[m][n])
                    else :
                        pass
                pwilist.append(pwi)
            
            print pwilist
            pwilistArray = np.array(pwilist)
            temppwilistArray = np.argsort(-pwilistArray)[0:10]
            ad = 0
            for i in range(0,10):
                ad += pwilistArray[temppwilistArray[i]]
            arrayd.append(ad/len(pwilist))
        fileJson['pwiMatrix'] = arrayd


        resultList.append(fileJson)
        # print fileJson
    reportJson['resultList'] = resultList
    json.dump(reportJson,reportFile)
    print reportJson

    print '正确个数'+str(acc)
    print len(file_list)
    correctrate=float(acc)/float(len(file_list))*100
    print correctrate
    e.set(str(correctrate))
    # print '正确率'+int(acc)/len(file_list)  #正确个数/总个数
    if len(file_list) ==1:
        original_e.set(file.split('/')[len(file.split('/'))-1].split('.')[0])
        flag_e.set(gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0])
    else:
        original_temp=''
        flag_temp=''
        original_e.set(original_temp)
        flag_e.set(flag_temp)    


# GUI

import Tkinter as tk
import tkMessageBox
import tkFileDialog
root= tk.Tk()
root.title("说话人识别")
root.geometry('395x235')
path1 = tk.StringVar()
path2 = tk.StringVar()
e = tk.StringVar()
original_e = tk.StringVar()
flag_e = tk.StringVar()
train_list = []
test_list = []

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
    train_path = path1.get().replace('u\'','').replace('\'','')
    global train_list
    train_path = train_path[1:-1]
    train_list = train_path.split(', ')
    if len(train_list) == 1:
        temp = train_list[0]
        temp = temp[0:-1]
        train_list[0] = temp
    tkMessageBox.showinfo(title='提示框', message='训练文件选择完成')
    # for i in range(0,len(train_list)):
    #     if i < len(train_list) or i ==len(train_list):
    #         print train_list[0]
    #     else:
    #         print 'end'

def selectPath2():
    path = tkFileDialog.askopenfilenames()
    path2.set(path)
    test_path = path2.get().replace('u\'','').replace('\'','')
    global test_list
    test_path = test_path[1:-1]
    test_list = test_path.split(', ')
    if len(test_list) == 1:
        temp = test_list[0]
        temp = temp[0:-1]
        test_list[0] = temp
    tkMessageBox.showinfo(title='提示框', message='测试文件选择完成')
    # for i in range(0,len(test_list)):
    #     if i < len(test_list) or i ==len(test_list):
    #         print test_list[i]
    #     else:
    #         print 'end'
# 训练
def train():
    # print 'train'
    # var1=entry1.get()
    train_GMM_N(train_list, 'models', np.int16)
    tkMessageBox.showinfo(title='提示框', message='训练完成')

# 测试
def test():
    print('test:')
    test_GMM_N(test_list, 'models', np.int16)
    tkMessageBox.showinfo(title='提示框', message='测试完成')
    # test_GMM_N('a/test_data', 'a/models', np.int16)




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

tk.Label(root,text = "  测试正确率:").place(x=5,y=165)
tk.Entry(root, textvariable =e,borderwidth = 3,width = 5,).place(x=85,y=165)
tk.Label(root,text = "  真实结果:").place(x=130,y=152)
tk.Entry(root, textvariable =original_e,borderwidth = 3,width = 22,).place(x=200,y=152)
tk.Label(root,text = "  测试结果:").place(x=130,y=182)
tk.Entry(root, textvariable =flag_e,borderwidth = 3,width = 22,).place(x=200,y=182)

if __name__ == '__main__':
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
