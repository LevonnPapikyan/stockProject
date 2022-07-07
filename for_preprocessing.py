import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from scipy.stats import linregress
import tensorflow
import numpy as np


def data_frame_to_numpy(data_frame):
    len_0 = len(data_frame)
    raw_data_f =np.array(data_frame)
    raw_data_f = raw_data_f.reshape(len_0, data_frame.shape[1])
    return raw_data_f

def split(data,n_steps, train_Length,label):
    test_length = 20
    inputdf = data
    #making our data to numpy array
    train_x = data_frame_to_numpy(  inputdf[-(n_steps+train_Length + test_length):-test_length]  )  
    train_y = (label[-(train_Length + test_length):-test_length]).to_numpy()
    test_x = data_frame_to_numpy(inputdf[-(test_length + n_steps):])
    test_y = label[-(test_length):].to_numpy()
    #makeing our final predictions
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(train_Length):
        x_train.append(train_x[i:(n_steps + i)])
        y_train.append(train_y[i])
    for i in range(test_length):
        x_test.append(test_x[i:(n_steps + i)])
        y_test.append(test_y[i])


    return np.array(x_train), np.array(y_train), np.array(x_test),np.array(y_test)


    
def mape( predictions, real):
        
    return sum(abs(predictions - real)/real)/len(real)

def mae( predictions, real):
    return sum(abs(predictions - real))/len(real)
