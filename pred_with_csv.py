from multiprocessing.dummy import Process
import numpy as np
import pandas as pd
from keras.models import load_model
from predictiontools import Prediction
import os
from predictiontools import mae, mape,plotting_final_high,plotting_final_low
import weights_change

def add_top_column(df, top_col, inplace=False):
    if not inplace:
        df = df.copy()
    df.columns = pd.MultiIndex.from_product([[top_col], df.columns])
    return df

    
stock_names = ["SOL"] #here you can add models names which you have trained
targets = ['LOW'] #here you can add HIGH
models = ["LSTM"] #here you can add WAVE XGB
models_folder = "models"
validation = 20


path_excel = "excel.xlsx"
writer = pd.ExcelWriter(path_excel)
for name in stock_names:
    list_for_models = []
    for j in models:
        for tar in targets:
            preprocces = pd.read_csv("preproccessing"+"_"+j+"_"+".csv")
            preprocces = preprocces.set_index("Unnamed: 0")

            if j=="LSTM":

                path_model = 'daily_models/'+ name
                scale_Y = preprocces.loc[name]['LABEL']
                scale_X = preprocces.loc[name]['DATA']
                multiply_log_X = preprocces.loc[name]['MULT']
                VOL = preprocces.loc[name]['VOL']
                df = pd.read_csv(path_model+".csv")
                df.DATE = pd.to_datetime(df.DATE.astype(str), format='%Y-%d-%m %H:%M:%S.%f', infer_datetime_format=True)   
                df = df.set_index("DATE")
                df['VOL'] = df['VOL']/int(VOL)
                lstm_model = load_model(os.path.join(models_folder, tar +"_"+ j+"_" + name + ".h5"))
                csv = Prediction(df,lstm_model,label_name=tar , scale_X=scale_X, validation_length =validation, scale_Y=scale_Y, multiply_log_X=multiply_log_X,visualize=False).to_csv()
                new_df = add_top_column(csv, tar+" "+j)
                list_for_models.append(new_df)
        
    
            elif j == "WAVE":
                if name == 'GAZP' and tar =='HIGH':
                    continue
                elif name =='MTSS' and tar == 'HIGH':
                    continue
                elif name == 'VTBR' and tar == 'HIGH':
                    continue
                else:
                    path_model = 'daily_models/'+ name
                    scale_Y = preprocces.loc[name]['LABEL']
                    scale_X = preprocces.loc[name]['DATA']
                    multiply_log_X = preprocces.loc[name]['MULT']
                    VOL = preprocces.loc[name]['VOL']
                    df = pd.read_csv(path_model+".csv")
                    df.DATE = pd.to_datetime(df.DATE.astype(str), format='%Y-%d-%m %H:%M:%S.%f', infer_datetime_format=True)   
                    df = df.set_index("DATE")
                    df['VOL'] = df['VOL']/int(VOL)
                    lstm_model = load_model(os.path.join(models_folder, tar  +"_"+ j+"_" + name + ".h5"))
                    csv = Prediction(df,lstm_model,label_name=tar , scale_X=scale_X, validation_length =validation, scale_Y=scale_Y, multiply_log_X=multiply_log_X,visualize=False).to_csv()
                    new_df = add_top_column(csv, tar+" "+j)
                    list_for_models.append(new_df)
                

    if (len(list_for_models)==3):
        df1 = list_for_models[0]
        df2 = list_for_models[1]
        df3 = list_for_models[2]
        df2.drop(("LOW LSTM",'OPEN'),inplace=True,axis=1)
        df3.drop(('LOW WAVE','OPEN'),inplace=True,axis=1)
        df3.drop(('LOW WAVE','REAL'),inplace=True,axis=1)
        c = pd.concat([df1,df2],join="inner",verify_integrity=True,axis=1)
        b = pd.concat([c,df3],axis=1)
        mae_lstm = mae(b.loc[:,("LOW LSTM","REAL")].values.squeeze()[:-1],b.loc[:,("LOW LSTM","PRED")].values.squeeze()[:-1])
        mae_wave = mae(b.loc[:,("LOW LSTM","REAL")].values.squeeze()[:-1],b.loc[:,"LOW WAVE"].values.squeeze()[:-1])
        sum_mae = mae_lstm+mae_wave
        b["FINAL LOW"] = ((mae_wave/(sum_mae))*b.loc[:,("LOW LSTM","PRED")].values.squeeze()+(mae_lstm/(sum_mae))*b.loc[:,"LOW WAVE"].values.squeeze())
        b.to_excel(writer, sheet_name = name)
        plotting_final_low(b,name)

    elif  (len(list_for_models)==4):

        df1 = list_for_models[0]
        df2 = list_for_models[1]
        df3 = list_for_models[2]
        df4 = list_for_models[3]
        df2.drop(("LOW LSTM",'OPEN'),inplace=True,axis=1)
        df3.drop(('HIGH WAVE','OPEN'),inplace=True,axis=1)
        df3.drop(('HIGH WAVE','REAL'),inplace=True,axis=1)
        df4.drop(("LOW WAVE",'OPEN'),inplace=True,axis=1)
        df4.drop(("LOW WAVE",'REAL'),inplace=True,axis=1)

        c = pd.concat([df1,df3],join="inner",verify_integrity=True,axis=1)

       
        mae_lstm = mae(c.loc[:,("HIGH LSTM","REAL")].values.squeeze()[:-1],c.loc[:,("HIGH LSTM","PRED")].values.squeeze()[:-1])
        mae_wave = mae(c.loc[:,("HIGH LSTM","REAL")].values.squeeze()[:-1],c.loc[:,"HIGH WAVE"].values.squeeze()[:-1])
        sum_mae = mae_lstm+mae_wave
        c["FINAL HIGH"] = ((mae_wave/(sum_mae))*c.loc[:,("HIGH LSTM","PRED")].values.squeeze()+(mae_lstm/(sum_mae))*c.loc[:,"HIGH WAVE"].values.squeeze())
        
        b = pd.concat([c,df2],axis=1)
        d = pd.concat([b,df4],axis=1)
        mae_lstm_l = mae(d.loc[:,("LOW LSTM","REAL")].values.squeeze()[:-1],d.loc[:,("LOW LSTM","PRED")].values.squeeze()[:-1])
        mae_wave_l = mae(d.loc[:,("LOW LSTM","REAL")].values.squeeze()[:-1],d.loc[:,"LOW WAVE"].values.squeeze()[:-1])
        sum_mae_l = mae_lstm_l+mae_wave_l
        d["FINAL LOW"] = ((mae_wave_l/(sum_mae_l))*d.loc[:,("LOW LSTM","PRED")].values.squeeze()+(mae_lstm_l/(sum_mae_l))*d.loc[:,"LOW WAVE"].values.squeeze())
        d.to_excel(writer, sheet_name = name)
        plotting_final_low(d,name)
        plotting_final_high(d,name)

writer.save()
print(12)

    
