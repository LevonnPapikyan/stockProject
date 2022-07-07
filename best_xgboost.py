import os
import pandas as pd
import numpy as np
from wandb.xgboost import wandb_callback
import matplotlib.pyplot as plt
import datetime
import wandb
from sklearn import preprocessing
import plotly
from for_preprocessing import data_frame_to_numpy, split
from model import createModel, LRLogger
from predictiontools import mae, mape,mse
import requests
import os 
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback
import pickle
import yfinance as yf
from yahoofinancials import YahooFinancials



stock_name = 'SOL'
project_name = stock_name +"  XGB"
n_steps_lstm = 15
n_steps_xgb = 50
train_length = 100
test_length =50
n_fetures= 5
label_name = 'LOW'


list_mae = []
n_steps= 50
train_Length = 100
def train_model():


  # Set default configurations (Defaults will be overwritten during sweep)
        config_defaults = {
        'max_depth': 50, 
        'num_leaves': 100,
        'n_estimators': 300,
        'learning_rate' : 0.001,
        'gamma' : 0.01,
        'min_child_weight' : 5,
        # 'ealy_stopping_rounds': 0.03
        }


        # Start W&B
        run = wandb.init(config=config_defaults)
        config = wandb.config
        # Load and split data

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

        # this column is going to be our label
        df["DIFFERENCE"] = df["High"]-df["Open"] 
        df["Open"]= df["Open"].shift(-1)
        label = df["DIFFERENCE"]
        df.drop("DIFFERENCE",axis=1,inplace=True)


        train_x, train_y, test_x, test_y = split(df,n_steps,train_Length,label)


        #scalling our data
        x_train = np.log(train_x).reshape(train_length*n_fetures,n_steps)
        y_train = train_y
        x_test = np.log(test_x).reshape(train_length*n_fetures,n_steps)
        y_test = test_y

        model = xgb.XGBRegressor(objective='reg:squarederror',gamma = config.gamma,
          learning_rate = config.learning_rate,max_depth = config.max_depth ,
          num_leaves = config.num_leaves
          ,n_estimators = config.n_estimators, min_child_weight = config.min_child_weight)
        # Fit regression model on train set
        
        model.fit(x_train, y_train)
        # Predict on test set
        y_preds = model.predict(x_test)
        # Evaluate predictions
        mae_score = mae(y_preds,y_test)

        list_mae.append(mae_score)

        mse_score = mse(y_preds,y_test)

        print(f"MAE: {round(mae_score, 4)}, MSE: {round(mse_score, 4)}")

        # Log model performance metrics to W&B

        wandb.log({"mae": mae_score, "mse": mse_score})

        if mae_score == min(list_mae):
            
            #fill and save results
            con = config._items.copy()
            del con['_wandb']
            con["n_steps"]=n_steps_xgb
            configuration = pd.DataFrame(con,index=[0])
            configuration.to_csv(stock_name+"_XGB_"+label_name+"_"+str(mae_score)+"_CONFIG.csv")

            real_test = y_test.squeeze()
            real_train = y_train.squeeze()
            
            test_pred = model.predict(x_test)
            test_base = y_test[-(test_length+1):-1]


            #making last predictions on train data
            train_pred = model.predict(x_train)
            train_base = y_train[-(len(y_train+1)):-(1)]

            #predecting mape for train
            mae_prediction_train = mae(train_pred.squeeze(),real_train)
            mae_base_train =mae(np.array(train_base.squeeze()),real_train[1:])

            #predecting mape for test
            mae_prediction_test = mae(np.array(test_pred.squeeze()),real_test)
            mae_base_test =mae(np.array(test_base.squeeze()),real_test[1:])
            #making series for plotting
            val_pred_series = pd.Series(data =test_pred.squeeze())
            val_test_real = pd.Series(data = real_test)
            train_pred_series = pd.Series(data =train_pred.squeeze())
            train_real_series = pd.Series(data = real_train)

            figname = 'xgb prediction' 
            #plotting our low and high columns
            plt.figure(figsize = (20,12))
            plt.plot([i for i in range(len(y_train))],train_pred_series, label = f"train predicted           MAE : {mae_prediction_train}", color = "red")
            plt.plot([i for i in range(len(y_train))],train_real_series, label = f"train predicted     base  MAE : {mae_base_train}", color = "green")
            plt.plot([i for i in range(len(y_train),len(y_train)+test_length)],val_pred_series, label = f"test predicted            MAE : {mae_prediction_test}", color = "blue")
            plt.plot([i for i in range(len(y_train),len(y_train)+test_length)],val_test_real, label =     f"test predicted      base  MAE :  {mae_base_test}", color = "black",linewidth =1)
            plt.legend(loc = 'upper left', fontsize = 15)
            plt.title("XGB "+stock_name+" LOW")
            plt.grid()

            wandb.log({figname:plt})

            plt.close()

sweep_configs = {
    "method": "random",
    "metric": {
        "name": "mse",
        "goal": "minimize"
    },
    "parameters": {

        "max_depth": {
            "values": [150,200,250,300,350,400,500,750,1000,1100,1200,1500]
        },

        "num_leaves": {
            "values": [50,70,150,180,200,300,500,700]
        },

        "n_estimators": {
            "values": [10,20,40,50,150,300,400,500,600,700]
        },

        "learning_rate": {
            "values" :[0.05,0.1,0.15,0.2,0.3,0.4,0.5]
        },

     "gamma": {
        "values" :[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.8]
     },
     "min_child_weight" : {
         
         "values" : [0,1,2,3,4,5,6,7,10,11,15,20,25,35,40,45,50,60,70,80]
     }

      
    }

}


sweep_id = wandb.sweep(sweep=sweep_configs, project=project_name)


wandb.agent(sweep_id=sweep_id, function=train_model, count=1000)
