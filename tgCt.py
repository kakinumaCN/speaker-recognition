import sys
import codecs
import os
import re
import wave
import numpy as np

if __name__ == '__main__':

    data_path = 'data1/'
    # train_data_path = 'a/train_data'
    # train_data_split_path = 'train_data_split'
    # test_data_path = 'a/test_data'
    data_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if not f.startswith('.')]
    print data_list

    for path in data_list:
        # file_name = os.listdir(path)[1].split('.')[0]
        file_names =[f.split('.')[0] for f in os.listdir(path) if f.endswith('.wav')]
        for file_name in file_names:
            file = codecs.open(os.path.join(path,file_name) + '.TextGrid', 'r', 'utf-16')
            # print file
            str_text = file.read()
            xmin_pattern = re.compile('xmin = (.*?) ', re.S)
            xmax_pattern = re.compile('xmax = (.*?) ', re.S)
            text_pattern = re.compile('text = (.*?) ', re.S)
            xmin_items = re.findall(xmin_pattern,str_text)
            xmax_items = re.findall(xmax_pattern,str_text)
            text_items = re.findall(text_pattern,str_text)

            f = wave.open(os.path.join(path,file_name) + '.wav',"rb")  
            params = f.getparams()  
            nchannels, sampwidth, framerate, nframes = params[:4]
            str_data  = f.readframes(nframes)  
            f.close()

            wave_data = np.fromstring(str_data,dtype = np.short)  
            wave_data.shape = -1,1  
            wave_data = wave_data.T  

            num = len(text_items)

            # print xmax_items
            # print xmin_items
            # print text_items

            for i in range(num):
                f_name = file_name + str(i) + text_items[i] +  '.wav'
                f_min = float(xmin_items[i+2])*framerate
                f_max = float(xmax_items[i+2])*framerate
                temp_data = wave_data[:,int(f_min):int(f_max)]

                f = wave.open(os.path.join('split',f_name), "wb")# 
                f.setnchannels(nchannels)
                f.setsampwidth(sampwidth)
                f.setframerate(framerate)
                f.writeframes(temp_data.tostring())
                f.close()  
