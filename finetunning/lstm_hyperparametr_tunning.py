import os 
import pandas as pd
import numpy as np
from wandb.integration.keras.keras import WandbCallback
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import json
import wandb
from sklearn import preprocessing
import plotly
from for_preprocessing import data_frame_to_numpy,split
from keras.models import load_model
from model import createModel, LRLogger
from predictiontools import mae, mape
import requests
import os 
import tensorflow as tf 
from keras.models import load_model
import yahoofinance as yf

#17:30
stock_name = 'SOL'

project_name = " hypertunning 11"

def train():
   
    
    configs = {

           "n_steps" : 15,
           'layers1': 128,
           'dropout1' : 0,
        #    'dropout2' : 0,
        #    'layers2' : 64,
           'layers3' : 32,
           'layers4' : 4,
           'reg1' : 0.002,
           'reg2' : 0.002,
           'reg3' : 0.002,
        #    'bath_size' : 16,
        #    'patience' : 10
        

       }

    run = wandb.init(project = project_name, config= configs)
    config = wandb.config


    
    n_steps = config.n_steps
    train_Length = 700
#config.patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.977,
                                patience=10, min_lr=0.000001) 


    l_r_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=2500,
        decay_rate=0.1)

   

    daily = yf.download('SOL-USD', 
                    #   start='2020-08-17', 
                        end='2022-05-20', 
                        progress=False,
    )



  
  
    daily["VOL"] = daily["VOL"]/10000
    df = daily.copy()
    datetimear = []
    for i in range(len(daily)):
            datetimear.append(daily.index[i])





    df["DIFFERENCE"] = df["OPEN"]-df["LOW"]   #adding new column which is the difference betwheen the open low
    df["OPEN"]= df["OPEN"].shift(-1)
    label = df["DIFFERENCE"]
    df.drop("DIFFERENCE",axis=1,inplace=True)


    train_x, train_y, test_x, test_y = split(df,n_steps,train_Length,label)


    x_train = np.log(train_x) #scalling our data
    y_train = train_y
    x_test = np.log(test_x)
    y_test = test_y



       # Initialize wandb with a sample project name


    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(config.layers1, return_sequences= True, activity_regularizer = tf.keras.regularizers.l1_l2(config.reg1)),
        tf.keras.layers.Dropout(config.dropout1),
        # Shape => [batch, time, features]
        # tf.keras.layers.LSTM(64,return_sequences= True), config.layers2
        tf.keras.layers.LSTM(64,activity_regularizer = tf.keras.regularizers.l1_l2(config.reg2)),
        # tf.keras.layers.Dropout(config.dropout2),
        tf.keras.layers.Dense(config.layers3, activity_regularizer = tf.keras.regularizers.l1_l2(config.reg3), activation='sigmoid'),
        tf.keras.layers.Dense(config.layers4, activation='tanh'),
        tf.keras.layers.Dense(units=1)])

       

       # Compile the model

    model.compile(optimizer='adam',

                     loss='mse',

                     metrics='mae')

       

       # Train the model   config.batch_size

    model.fit(x_train, y_train, batch_size = 16, epochs=300,

                     validation_data=(x_test, y_test),

                callbacks=[WandbCallback(monitor= 'val_loss', save_model= True),reduce_lr, LRLogger(model.optimizer)])



sweep_config = {

       'method': 'grid',

       'parameters': {

           'n_steps' : {

               'values' : [15,20,25,40,50]
           },

           'layers1': {

                  'values': [ 96, 128, 256]},
            
        #    'layers2': {

        #           'values': [32, 64, 96, 128, 256]},
            
            'layers3': {

                  'values': [32, 64]},
            
             'layers4': {

                  'values': [8,16,32]},
            
            'dropout1' : {

                'values' : [0,0.5,0.2,0.4,0.7]
            },
            # 'dropout2' : {

            #     'values' : [0,0.5,0.2,0.4,0.7]
            # },
            'reg1' : {
                'values' : [0.2,0.1,0.3,0.005,0.01,0.7]
            },
            'reg2' : {
                'values' : [0.002,0.0003,0.01,0.7]
            },
            'reg3' : {
                'values' : [0.1,0.2,0.3,0.01,0.7,0.003]
            },
            
            # 'batch_size' : {

            #      'values' : [16,32,64,128,256,521]  },
            
            
            # 'patience' : {

            #     'values' : [5,10,15,20]
            # }
        
        

       }

    }


sweep_id = wandb.sweep(sweep_config,project=project_name) 


wandb.agent(sweep_id, function=train,count=25)