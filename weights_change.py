import os 
import pandas as pd
import numpy as np
from wandb.integration.keras.keras import WandbCallback
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import wandb
from sklearn import preprocessing
import plotly
from for_preprocessing import data_frame_to_numpy,split,mae,mape
from keras.models import load_model
from model import createModel, LRLogger
import matplotlib.pyplot as plt
import yfinance as yf
from yahoofinancials import YahooFinancials
from pred_with_csv import stock_names,targets

stock_name = stock_names[0]
target = targets[0]
n_steps = 15
train_Length =200


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.977,
                              patience=10, min_lr=0.000001) 


l_r_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=2500,
    decay_rate=0.1)


project_name = stock_name + " experiments"


daily = yf.download('SOL-USD', 
                    #   start='2020-08-17', 
                      end='2022-05-20', 
                      progress=False,
)


df = daily.copy()
datetimear = []
for i in range(len(df)):
    datetimear.append(df.index[i])


df["DIFFERENCE"] = df["High"]-df["Open"]  # this column is going to be our label
df["Open"]= df["Open"].shift(-1)
label = df["DIFFERENCE"]
df.drop("DIFFERENCE",axis=1,inplace=True)


train_x, train_y, test_x, test_y = split(df,n_steps,train_Length,label)


x_train = np.log(train_x) #scalling our data
y_train = train_y
x_test = np.log(test_x)
y_test = test_y


n_epochs = 5
batch_size = 64


models_list = []
model_number = []
val_los_list=[]
all_list = []
best_model_val_los = []


for i in range(3):
    model = createModel()
    model.compile(loss="mse", optimizer='adam',
               metrics=["mae"])
    run = wandb.init(project = project_name,reinit = True,config=tf.compat.v1.flags,  name = "model_num:" +  str(i + 1) )
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs = n_epochs,
                         validation_data=(x_test, y_test),  callbacks=[WandbCallback(monitor= 'val_loss', save_model= True),reduce_lr, LRLogger(model.optimizer)] )
    model_best = load_model((wandb.run.dir+str("\\model-best.h5")).replace("\\", "/"))
    models_list.append(model_best)
    model_number.append(i)
    val_los_list.append(history.history["val_loss"]) 
    best_model_val_los.append(min(val_los_list[i]))

run.finish()
zipped_list = sorted(zip(best_model_val_los, models_list, model_number,val_los_list))
test_length = 20
real_test = label[-20:].squeeze()
real_train = label[-(train_Length + test_length):-(test_length)]

model = zipped_list[0][1]
test_pred = model.predict(x_test)
test_base = label[-(test_length+1):-1]

#making last predictions on train data
train_pred = model.predict(x_train)
train_base = label[-(train_Length + test_length+1):-(test_length+1)]

#predecting mape for train
mae_prediction_train = mae(train_pred.squeeze(),real_train)
mae_base_train =mae(np.array(train_base.squeeze()),real_train)

#predecting mape for test
mae_prediction_test = mae(np.array(test_pred.squeeze()),real_test)
mae_base_test =mae(np.array(test_base.squeeze()),real_test)

#making series for plotting
val_pred_series = pd.Series(data =test_pred.squeeze())
val_test_real = pd.Series(data = real_test)
train_pred_series = pd.Series(data =train_pred.squeeze())
train_real_series = pd.Series(data = real_train)
figname = 'model ' + str(zipped_list[0][2])

#plotting our low and High columns
plt.figure(figsize = (20,12))
plt.plot(datetimear[-(train_Length + test_length):-(test_length)],train_pred_series, label = f"train predicted           MAE : {mae_prediction_train}", color = "r")
plt.plot(datetimear[-(train_Length + test_length):-(test_length)],train_real_series, label = f"train predicted     base  MAE : {mae_base_train}", color = "green")
plt.plot(datetimear[-(test_length):],val_pred_series, label =   f"test predicted            MAE : {mae_prediction_test}", color = "blue")
plt.plot(datetimear[-(test_length):],val_test_real, label =     f"test predicted      base  MAE :  {mae_base_test}", color = "black",linewidth =1)
plt.legend(loc = 'upper left', fontsize = 15)
plt.title("LSTM SOL High")
plt.grid()
plt.show()
plt.close()
plt.plot(zipped_list[0][3], label = "model number:" + str(zipped_list[0][2]))
plt.plot(zipped_list[1][3], label = "model number:" + str(zipped_list[1][2]))
plt.plot(zipped_list[2][3], label = "model number:" + str(zipped_list[2][2]))
plt.title("losses")
plt.show()
plt.close()
model.save(stock_name + "_model_the_BEST"+".h5")




    