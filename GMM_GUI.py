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
correctrate=float()

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_MFCC(sr, audio):
    return preprocessing.scale(mfcc(audio, sr,  appendEnergy=False))

import random
def getRondomList(len_feature):
    i = 0
    random_list = []
    while i < int(len_feature*0.8):
        temp = random.randint(0, len_feature-1)
        if temp in random_list:
            continue
        else:
            random_list.append(temp)
            i = i + 1
    return random_list

def train_GMM_N(speaker_list, model_path, dtype):
    print speaker_list,model_path
    N = len(speaker_list)
    train_path = [0 for i in range(N)]
    train_mfcc_features = [0 for i in range(N)]
    speaker_gmm = [0 for i in range(N)]
    for i in range(N):
        train_mfcc_features[i] = np.asarray((), dtype=dtype)
        f = wave.open(speaker_list[i], 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)

        tfeature = get_MFCC(frame_rate, audio)

        # feature = []
        # random_list = getRondomList(len(tfeature))
        # for r in random_list:
        #     feature.append(tfeature[r])

        # print i,len(tfeature),len(feature),len(random_list)

        if train_mfcc_features[i].size == 0:
            train_mfcc_features[i] = tfeature
        else:
            train_mfcc_features[i] = np.vstack((train_mfcc_features[i], tfeature))
            print 'vstack'
    
        # speaker_gmm[i] = GMM(n_components=4, n_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200,covariance_type='diag', n_init=3)
        speaker_gmm[i].fit(train_mfcc_features[i])
        print speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]
        pickle.dump(speaker_gmm[i], open(os.path.join(model_path, speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]+'.gmm'), 'w'))    
 
def test_GMM_N(file_list, model_path, dtype):
    gmm_files = []
    for f in os.listdir(model_path):
        if f.endswith('.gmm'):
            gmm_files.append(model_path+'/'+f)
    models = [pickle.load(open(f, 'r')) for f in gmm_files]
    result = []
    acc = 0
    reportJson = {}
    resultList = []
    reportJson['resultList'] = resultList
    for file in file_list:
        print file
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            scores = np.array(models[i].score(feature))
            log_likelihood[i] = scores.sum()
        if file.split('/')[len(file.split('/'))-1].split('.')[0] ==gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0]:
            acc += 1
        else:
            pass
        
        fileJson = {}
        fileJson['testFilename'] = file
        loglike = []
        for log in log_likelihood:
            loglike.append(log)

        loglikeArray = np.array(loglike)
        minloglike = np.argsort(-loglikeArray)[0:3]
        arraya = []
        arrayb = []
        arrayc = []
        for i in range(0,3):
            arraya.append(loglike[minloglike[i]])
            arrayb.append(gmm_files[minloglike[i]])
            arrayc.append(models[minloglike[i]])
        fileJson['likelihoodList'] = arraya
        fileJson['modelList'] = arrayb


        resultList.append(fileJson)
        reportFile = codecs.open(os.path.join(model_path,'gmm_report_likehood.json'), 'w', 'utf-8')
        json.dump(reportJson,reportFile)

    # for file in file_list:
    # #     fileJson = {}
    # #     fileJson['testFilename'] = file
    # #     fileJson['modelList'] = gmm_files
    # #     loglike = []
    # #     for log in log_likelihood:
    # #         loglike.append(log)
    # #     fileJson['likelihoodList'] = loglike

    # #     loglikeArray = np.array(loglike)
    # #     minloglike = np.argsort(-loglikeArray)[0:3]
    # #     arraya = []
    # #     arrayb = []
    # #     arrayc = []
    # #     for i in range(0,3):
    # #         arraya.append(loglike[minloglike[i]])
    # #         arrayb.append(gmm_files[minloglike[i]])
    # #         arrayc.append(models[minloglike[i]])
    # #     fileJson['likelihoodList'] = arraya
    # #     fileJson['modelList'] = arrayb
 
    #     arrayd = []
    #     print '2'
    #     for i in range(len(arrayc)):
    #         print 'i' + str(i)
    #         pwilist = []

    #         if len(arrayc[i]._estimate_weighted_log_prob(feature)) > 1300:
    #             for m in range(1250,1260):
    #                 print 'm' + str(m)
    #                 #each sample
    #                 pwi = 0
    #                 for n in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature)[m])):
    #                     # print (arrayc[i]._estimate_weighted_log_prob(feature)[m])
    #                     #each component
    #                     if arrayc[i]._estimate_weighted_log_prob(feature)[m][n] > -9:
    #                             pwi += math.exp(arrayc[i]._estimate_weighted_log_prob(feature)[m][n])
    #                     else :
    #                         pass
    #                 pwilist.append(pwi)

    #         elif len(arrayc[i]._estimate_weighted_log_prob(feature)) > 800:
    #             for m in range(0,800): 
    #                 #each sample
    #                 pwi = 0
    #                 for n in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature)[m])):
    #                     #each component
    #                     if arrayc[i]._estimate_weighted_log_prob(feature)[m][n] > -9:
    #                             pwi += math.exp(arrayc[i]._estimate_weighted_log_prob(feature)[m][n])
    #                     else :
    #                         pass
    #                 pwilist.append(pwi)
    #         else:
    #             for m in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature))): 
    #                 #each sample345
    #                 pwi = 0
    #                 for n in range(0,len(arrayc[i]._estimate_weighted_log_prob(feature)[m])):
    #                     #each component
    #                     if arrayc[i]._estimate_weighted_log_prob(feature)[m][n] > -9:
    #                             pwi += math.exp(arrayc[i]._estimate_weighted_log_prob(feature)[m][n])
    #                     else :
    #                         pass
    #                 pwilist.append(pwi)
    #         pwilistArray = np.array(pwilist)
    #         temppwilistArray = np.argsort(-pwilistArray)[0:10]
    #         ad = 0
    #         for i in range(0,10):
    #             ad += pwilistArray[temppwilistArray[i]]
    #         arrayd.append(ad/len(pwilist))
    #     fileJson['pwiMatrix'] = arrayd

    #     print '3'
    #     resultList.append(fileJson)
    #     reportFile = codecs.open('gmm_report_pwi.json', 'w', 'utf-8')
    #     json.dump(reportJson,reportFile)

    
    print '正确个数'+str(acc)
    print len(file_list)
    correctrate=float(acc)/float(len(file_list))*100
    print correctrate
    e.set(str(correctrate))
    if len(file_list) ==1:
        original_e.set(file.split('/')[len(file.split('/'))-1].split('.')[0])
        flag_e.set(gmm_files[np.argmax(log_likelihood)].split('/')[len(gmm_files[np.argmax(log_likelihood)].split('/'))-1].split('.')[0])
    else:
        original_temp=''
        flag_temp=''
        original_e.set(original_temp)
        flag_e.set(flag_temp)    

    # # import json
    # import xlwt
    # # import codecs
    # workbook = xlwt.Workbook(encoding = 'ascii')
    # worksheet = workbook.add_sheet('sheet_1')
    # worksheet.write(0,0,label = 'fileName')
    # worksheet.write(0,1,label = '1st')
    # worksheet.write(0,2,label = '1st_likehood')
    # worksheet.write(0,3,label = '2nd')
    # worksheet.write(0,4,label = '2nd_likehood')
    # worksheet.write(0,5,label = '3rd')
    # worksheet.write(0,6,label = '3rd_likehood')
    # f = codecs.open('gmm_report.json','rb')
    # j = json.load(f)
    # c = 1
    # for result in j['resultList']:
    #     # print result
    #     worksheet.write(c,0,label = result['testFilename'])
    #     worksheet.write(c,1,label = result['modelList'][0])
    #     worksheet.write(c,2,label = result['likelihoodList'][0])
    #     worksheet.write(c,3,label = result['modelList'][1])
    #     worksheet.write(c,4,label = result['likelihoodList'][1])
    #     worksheet.write(c,5,label = result['modelList'][2])
    #     worksheet.write(c,6,label = result['likelihoodList'][2])
    #     c += 1
    # workbook.save('gmm_report_likehood.xls')

def train_GMM_N_80(speaker_list, model_path, dtype):
    time = 3
    print speaker_list,model_path
    N = len(speaker_list)
    train_path = [0 for i in range(N)]
    train_mfcc_features = [0 for i in range(N)]
    speaker_gmm = [0 for i in range(N)]
    for i in range(N):
        f = wave.open(speaker_list[i], 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        tfeature = get_MFCC(frame_rate, audio)

        for t in range(time):
            train_mfcc_features[i] = np.asarray((), dtype=dtype)
            feature = []
            random_list = getRondomList(len(tfeature))
            for r in random_list:
                feature.append(tfeature[r])

            print i,t,len(tfeature),len(feature)

            if train_mfcc_features[i].size == 0:
                train_mfcc_features[i] = feature
            else:
                train_mfcc_features[i] = np.vstack((train_mfcc_features[i], feature))
                print 'vstack'
        
            # speaker_gmm[i] = GMM(n_components=4, n_iter=200,covariance_type='diag', n_init=3)
            speaker_gmm[i] = GaussianMixture(n_components=32, max_iter=200,covariance_type='diag', n_init=3)
            speaker_gmm[i].fit(train_mfcc_features[i])
            print speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]
            pickle.dump(speaker_gmm[i], open(os.path.join(model_path+'/'+str(t), speaker_list[i].split('/')[len(speaker_list[i].split('/'))-1].split('.')[0]+'.gmm'), 'w'))    

def test_GMM_N_80(file_list, model_path, dtype): 
    time = 3
    for t in range(time):
        print 'time',t
        test_GMM_N(file_list, model_path+'/'+str(t), dtype)

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
    path = tkFileDialog.askopenfilenames()
    path1.set(path)
    global train_list
    if str(path).find("(") >= 0:
        train_path = path1.get().replace('u\'','').replace('\'','')
        train_path = train_path[1:-1]
        train_list = train_path.split(', ')
        if len(train_list) == 1:
            temp = train_list[0]
            temp = temp[0:-1]
            train_list[0] = temp
    else:
        train_list = str(path).split(' ')
    tkMessageBox.showinfo(title='提示框', message='训练文件选择完成')

def selectPath2():
    path = tkFileDialog.askopenfilenames()
    print('select:')
    print path
    print('_____')
    path2.set(path)
    global test_list
    if str(path).find("(") >= 0:
        test_path = path2.get().replace('u\'','').replace('\'','')
        test_path = test_path[1:-1]
        test_list = test_path.split(', ')
        if len(test_list) == 1:
            temp = test_list[0]
            temp = temp[0:-1]
            test_list[0] = temp
    else:
        test_list = str(path).split(' ')    
    tkMessageBox.showinfo(title='提示框', message='测试文件选择完成')

def train():
    train_GMM_N_80(train_list, 'models9', np.int16)
    tkMessageBox.showinfo(title='提示框', message='训练完成')

def test():
    print('testlist:')
    print test_list
    print('_____')
    test_GMM_N_80(test_list, 'models9', np.int16)
    tkMessageBox.showinfo(title='提示框', message='测试完成')


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
    # test_GMM_N(['/Users/suntian/Desktop/NHv2.0/NHtestset-2/shibie-1/20170825113532_1_0_C.wav','/Users/suntian/Desktop/NHv2.0/NHtestset-2/shibie-1/20170817141025_2_2_L.wav'], 'models', np.int16)
    # train_GMM_N(['/Users/suntian/Developer/VPR/dataSet/testset6_after_ring/zhuce/20170821134345_1_3_L.wav', '/Users/suntian/Developer/VPR/dataSet/testset6_after_ring/zhuce/20170822014423_2_0_L.wav'], 'models3', np.int16)
