import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import linregress
import wandb

class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
      super(LRLogger, self).__init__()
      self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
      lr = self.optimizer.learning_rate.numpy()
      wandb.log({"learning rate": lr}, commit=False)            

def createModel():
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(128, return_sequences= True, activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-4,l2=1e-4)),
        # tf.keras.layers.Dropout(0.3),
        # Shape => [batch, time, features]
        # tf.keras.layers.LSTM(64,return_sequences= True),
        tf.keras.layers.LSTM(64,activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-4,l2=1e-4)),
        tf.keras.layers.Dense(units=16, activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-4,l2=1e-4), activation='sigmoid'),
        tf.keras.layers.Dense(units=4, activation='tanh'),
        tf.keras.layers.Dense(units=1)])
        
    return model
