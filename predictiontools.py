
import datetime
from operator import mul
from sklearn import preprocessing
import plotly
import pandas as pd
import tensorflow
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import keras
import xgboost as xgb
class Prediction:
                                                                               
    """
    A class to make a prediction on stock price.

    Attributes
    -------

    daily : DataFrame
         Data with which the predction will be done

    model : h5 file       
        Trained model by which the prediction will be done

    label_name : str   
        The label which needs to  be predicted

    scale_X : float or int
        The number by which the model have been trained
    
    scale_Y : float or int 
    The number by which the model have been trained
    
    multiply_log_x : float or int
    The number by which the model have been trained

    visualize: boolean
    Whether user wants to see the visualization of prediction or not

    validation_length : int
        The number of days you want to predict
    
    -------


    Methods
    -------
    labelling(label_name, df):
        Divides data into two parts label and data without label
    
    reshaping(df, label, n_steps, scale_X):
        Prepairs label and data for prediction and further calculations
    
    prediction(): 
        Makes first prediction on data
    
    final_prediction():
        Counts final prediction 
    
    _mae_mape_for_final_pred_with_base():
        Counts mean absalute error and mean absalute percantage error for final prediction

    
    mape(), mae(), mse():  
        Counts mean absalute percentage error, mean absalute error, mean squeared error

    data_frame_to_numpy():
        Makes dataframe into an numpy array
    
    _plotting():
        Visualizing final prediction
    
    plotting_difference():
        Visualizing difference (main prediction)
    
    to_csv():
        Makes a csv file with predictions


    -------
    """                                                        
                                              
    
    def __init__(self, daily, model, label_name, scale_X, validation_length=50, scale_Y = None,multiply_log_X = 1, visualize = False,n_fetures=5): 
         
        """
    Constructs all the necessary attributes for the Prediction object.

    Parameters
    ----------
    daily : Pandas DataFrame
    Atribute which helps for further calcuations
    
    n_steps : int 
    Number of days by whcih the predction is held

    label_name : str
    Name of the label needs to be predicted

    model : keras.engine.sequential.Sequential
    trained LSTM XGB or WAVENET model


        """

        self.visualize = visualize
        self.multiply_log_X = multiply_log_X
        self.scale_Y = scale_Y           
        self.daily = daily.copy()  
        self.label_name = label_name 
        self.scale_X = scale_X
        self.model = model
        self.validation_length = validation_length  
        self.final_prediction_base = None 
        self.mae_prediction_test = None  
        self.mae_base_test = None  
        self._datetimear = []  
        
        if type(self.model)!=keras.engine.sequential.Sequential:

             self.n_steps = int(self.model.n_features_in_/n_fetures)

        
        else:

            self.n_steps = self.model.get_config()["layers"][0]["config"]["batch_input_shape"][1] 


        if len(daily[-(validation_length+self.n_steps):])<= self.n_steps:
             
            raise ValueError("sizes of data and number of steps don't match together")

        
        for i in range(len(self.daily)):

            self._datetimear.append(self.daily.index[i])
                

        self.df, self.label = self.labeling(label_name=self.label_name, df = daily[-(validation_length+self.n_steps):] )


        """
        this part of class is checking how to scale data and label:

        scale_Y: 'log' or integer

        scale_X: 'log' or integer

        when we log label we add one in case of not to haveing 0.
        we also multiply our data while counting logarithm of it.
         
        """
        
        if self.scale_Y == 'log' and isinstance(eval(self.scale_X),(float,int))== True:
                    
            self.df_scaled, self.label_scaled = self.df/float(self.scale_X), np.log(self.label+1)

        elif self.scale_Y == 'log' and self.scale_X == 'log':
                
            self.df_scaled, self.label_scaled = np.log(self.df*float(multiply_log_X)), np.log(self.label+1)

        elif isinstance(eval(str(self.scale_Y)),(float,int))==True and self.scale_X=='log':

            self.df_scaled, self.label_scaled = np.log(self.df*float(self.multiply_log_X)), self.label/float(self.scale_Y)

        elif isinstance(eval(str(self.scale_Y)),(float,int))==True and isinstance(eval(self.scale_X),(float,int))== True:

            self.df_scaled, self.label_scaled = self.df/float(self.scale_X), self.label/float(self.scale_Y)


        self.x_data, self.y_data = self.reshaping(df= self.df_scaled, label = self.label_scaled, n_steps=self.n_steps)
        
        self.difference_prediction = self.prediction( model = self.model, x_data=self.x_data)
    
        
        
        if self.scale_Y == 'log':
             
            self.difference_prediction = np.exp(self.difference_prediction)-1
             
        else :
            
            self.difference_prediction = self.difference_prediction*float(self.scale_Y)


        self.final_prediction = self.final_predictions( daily = self.daily,label_name=self.label_name, validation_length = self.validation_length, difference_prediction=self.difference_prediction)
        
        if visualize==True:

            self.mae_prediction_test , self.mae_base_test, self.mape_base_test, self.mape_prediction_test = self._mae_mape_for_final_pred_with_base()
            self.mae_mse_difference()
            self._plotting()
            self.plotting_difference()

   
    def mae_mse_difference(self):
        
        """
        plotting results of your prediction
        mae mse 

        """
        print("mae prediction ", self.mae(np.array(self.difference_prediction.squeeze()),np.array(self.label[-self.validation_length:])))     
        print('mae base ',self.mae(np.array(self.label[-(self.validation_length + 1):-1]),np.array(self.label[-self.validation_length:])))
        print("mse prediction ", self.mse(np.array(self.difference_prediction.squeeze()),np.array(self.label[-self.validation_length:])))    
        print('mse base ',self.mse(np.array(self.label[-(self.validation_length + 1):-1]),np.array(self.label[-self.validation_length:])))
        

    def labeling(self, label_name, df):
    
        if label_name == 'LOW':
            print("df len : ", len(df))
            
            df["DIFFERENCE"] = df["OPEN"]-df["LOW"]   #adding new column which is the difference betwheen the OPEN and LOW
            df["OPEN"] = df["OPEN"].shift(-1)
            label = df["DIFFERENCE"]
            df.drop("DIFFERENCE",axis=1,inplace=True)

            return df, label

        else:

            df["DIFFERENCE"] =df["HIGH"] - df["OPEN"]  #adding new column which is the difference betwheen the HIGH and OPEN
            df["OPEN"] = df["OPEN"].shift(-1)
            label = df["DIFFERENCE"]
            df = df.drop("DIFFERENCE",axis=1)

            return df, label


    def reshaping(self, df, label, n_steps,n_fetures=5):  #function for preproccesing
            x_data = []
            y_data = []
            inputdf = df
            data_Length = len(df)-n_steps
            

            #making our data to numpy array
            data_x = self.data_frame_to_numpy(  inputdf[-(n_steps+data_Length ):])  
            data_y = (label[-(data_Length ):]).to_numpy()

            #makeing our final predictions
            for i in range(data_Length):
                x_data.append(data_x[i:(n_steps + i)])
                y_data.append(data_y[i])

            # x_data = np.array(x_data )/scale_X  #scale_X our data
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            if type(self.model)!=keras.engine.sequential.Sequential:

                 x_data = np.array(x_data).reshape(data_Length,n_fetures*n_steps)
                 y_data = np.array(y_data)

                
            return x_data, y_data

    def prediction(self, model, x_data):

        difference_prediction = model.predict(x_data)

        return difference_prediction
    
    def final_predictions(self , daily, label_name, validation_length, difference_prediction ): #for final prediction we need to have real data

    #slicing open column 
        _open_resized_for_data = daily["OPEN"][-(validation_length):]

        if label_name == 'HIGH':
            final_prediction =  _open_resized_for_data.values + difference_prediction.squeeze()
            
            return final_prediction

        elif label_name == 'LOW':
            final_prediction =  _open_resized_for_data.values - difference_prediction.squeeze()
            
            return final_prediction
    
    def _mae_mape_for_final_pred_with_base(self):

        open_resized_for_data = self.daily["OPEN"][-(self.validation_length):] 

        if self.label_name == 'LOW':
            #for test data base mape
            yesterday_data = np.array(self.daily[-self.validation_length-1:][self.label_name])
            _df = np.array(self.daily.iloc[-self.validation_length-1:,0])
            suma =  _df - yesterday_data
            suma = np.array(suma[:-1])
            self.final_prediction_base = open_resized_for_data - suma    
            self.mae_prediction_test = self.mae(np.array(self.final_prediction).squeeze(),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mae_base_test =self.mae(np.array(self.final_prediction_base),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mape_prediction_test = self.mape(np.array(self.final_prediction).squeeze(),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mape_base_test =self.mape(np.array(self.final_prediction_base),np.array(self.daily[self.label_name][(-self.validation_length):]))

            return  self.mae_prediction_test , self.mae_base_test, self.mape_base_test, self.mape_prediction_test
        
        if self.label_name == 'HIGH':
            yesterday_data = np.array(self.daily[-self.validation_length-1:][self.label_name])
            _df = np.array(self.daily.iloc[-self.validation_length-1:,0])
            suma = yesterday_data - _df
            suma = np.array(suma[:-1])
            self.final_prediction_base = open_resized_for_data + suma    
            self.mae_prediction_test = self.mae(np.array(self.final_prediction).squeeze(),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mae_base_test =self.mae(np.array(self.final_prediction_base),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mape_prediction_test = self.mape(np.array(self.final_prediction).squeeze(),np.array(self.daily[self.label_name][(-self.validation_length):]))
            self.mape_base_test =self.mape(np.array(self.final_prediction_base),np.array(self.daily[self.label_name][(-self.validation_length):]))


            return  self.mae_prediction_test , self.mae_base_test, self.mape_base_test, self.mape_prediction_test

    def _plotting(self):

        if self.label_name == 'LOW':
            yesterday_data = np.array(self.daily[-self.validation_length-1:][self.label_name])
            _df = np.array(self.daily.iloc[-self.validation_length-1:,0])
            suma = yesterday_data - _df
            suma = np.array(suma[:-1])
            self.final_prediction_base = self.daily["OPEN"][-(self.validation_length):]  + suma 

            pred_series = pd.Series(data =self.final_prediction)
            val_real = pd.Series(data =  np.array(self.daily[self.label_name][-self.validation_length:] ))
    
            plt.figure(figsize = (20,12))
            plt.plot(self._datetimear[-self.validation_length :],np.array(pred_series),label = f"prediction MAE : {self.mae_prediction_test}")
            plt.plot(self._datetimear[-self.validation_length :],self.final_prediction_base, label = f"                  BASE MAE :  {self.mae_base_test}", color = "black",linewidth =1)
            plt.plot(self._datetimear[-self.validation_length :],val_real,label = '                  real data ' ,c = 'g',linewidth =0.5)
            plt.grid()
            plt.legend(loc = 'upper left',fontsize = 15)
            plt.title("prediction")
            plt.show()

        if self.label_name == 'HIGH':

            pred_series = pd.Series(data =self.final_prediction)
            val_real = pd.Series(data =  np.array(self.daily[self.label_name][-self.validation_length:] ))
            
            plt.figure(figsize = (20,12))
            plt.plot(self._datetimear[-self.validation_length :],np.array(pred_series),label = f" prediction MAE : {self.mae_prediction_test}")
            plt.plot(self._datetimear[-self.validation_length :],self.final_prediction_base, label = f"                  BASE MAE :  {self.mae_base_test}", color = "black",linewidth =0.5)
            plt.plot(self._datetimear[-self.validation_length :],val_real,label = '                     real data',c = 'g', linewidth =0.5)
            plt.grid()
            plt.legend(loc = 'upper left',fontsize = 15)
            plt.title("prediction")
            plt.show()

    def plotting_difference(self): 

            pred_diff = pd.Series(data =self.difference_prediction.squeeze() )
            val_real = pd.Series(data =  np.array(self.label[-self.validation_length:] ))
            pred_base = pd.Series(data = np.array(self.label[-(self.validation_length+1):-1] ))
            
            plt.figure(figsize = (20,12))
            plt.plot(self._datetimear[-self.validation_length :],np.array(pred_diff),label = f" prediction MAE : {self.mae_prediction_test}")
            plt.plot(self._datetimear[-self.validation_length :],pred_base, label = f"                  BASE MAE :  {self.mae_base_test}", color = "black",linewidth =1)
            plt.plot(self._datetimear[-self.validation_length :],val_real,label = '                   real data',c = 'g',linewidth =0.5)
            plt.grid()
            plt.legend(loc = 'upper left',fontsize = 15)
            plt.title("prediction")
            plt.show()
            
    def to_csv(self):
      
      val_real_final = pd.Series(data =  np.array(self.daily[self.label_name][-self.validation_length:-1]),index = self._datetimear[-self.validation_length :-1])
      val_prediction_final = pd.Series(data =  self.final_prediction,index = self._datetimear[-self.validation_length :])
      open = pd.Series(data = self.daily['OPEN'][-self.validation_length:])
     
      dict1 = {'OPEN' : open , 'REAL' : val_real_final, "PRED" : val_prediction_final}
      data_csv = pd.DataFrame(data = dict1, index = self._datetimear[-self.validation_length :] )
      return data_csv
     
    def mape(self, predictions, real):
        
        return sum(abs(predictions - real)/real)/len(real)

    def mae(self, predictions, real):
        return sum(abs(predictions - real))/len(real)

    def mse(self,predictions, real):
        return sum((predictions - real)**2)/len(real)

    def data_frame_to_numpy(self,data_frame):
        len_0 = len(data_frame)
        raw_data_f =np.array(data_frame)
        raw_data_f = raw_data_f.reshape(len_0, data_frame.shape[1])
        
        return raw_data_f
    

def mape( predictions, real):
        
    return sum(abs(predictions - real)/real)/len(real)

def mae(predictions, real):
    return sum(abs(predictions - real))/len(real)

def mse(predictions, real):
    return sum((predictions - real)**2)/len(real)

       
def plotting_final_low(df,name):
    real = df.loc[:, ("LOW LSTM",'REAL')]
    lstm = df.loc[:, ('LOW LSTM','PRED')]
    wave = df.loc[:, ('LOW WAVE','PRED')]
    together = df.loc[:,('FINAL LOW')]
    plt.figure(figsize = (20,12))
    plt.plot(real,label = "REAL")
    plt.plot(lstm, label = "LSTM", color = "black",linewidth =1)
    plt.plot(wave,label = 'WAVENET',c = 'g',linewidth =1)
    plt.plot(together,label='TOGETHER')
    plt.grid()
    plt.legend(loc = 'upper left',fontsize = 15)
    plt.title("prediction "+name)
    plt.show()
       
def plotting_final_high(df,name):
    real = df.loc[:, ("HIGH LSTM",'REAL')]
    lstm = df.loc[:, ('HIGH LSTM','PRED')]
    wave = df.loc[:, ('HIGH WAVE','PRED')]
    together = df.loc[:,('FINAL HIGH')]
    plt.figure(figsize = (20,12))
    plt.plot(real,label = "REAL")
    plt.plot(lstm, label = "LSTM", color = "black",linewidth =1)
    plt.plot(wave,label = 'WAVENET',c = 'g',linewidth =1)
    plt.plot(together,label='TOGETHER')
    plt.grid()
    plt.legend(loc = 'upper left',fontsize = 15)
    plt.title("prediction "+name)
    plt.show()



def data_frame_to_numpy(self,data_frame):
     len_0 = len(data_frame)
     raw_data_f =np.array(data_frame)
     raw_data_f = raw_data_f.reshape(len_0, data_frame.shape[1])
     
     return raw_data_f
 
class XgbTrain:
    """
    this class is for makeing Xgboost training 
    methods
    -------
    """
    def __init__(self,train_df,params,label_name,train_length=800):
        """
          Constructs all the necessary attributes for the Prediction object.
    Parameters
    ----------
        train_df : pandas DataFrame
        
        contains training data
        params : dict
        contains the configurations by which model has already been trained
        """    
        self.train_df = train_df
        self.n_steps = int(params["n_steps"])
        self.params = params
        self.params.pop("n_steps")
        self.train_length = train_length
        self.label_name = label_name
    
    def labeling(self, label_name, df):
    
        if label_name == 'LOW':
            
            df["DIFFERENCE"] = df["OPEN"]-df["LOW"]   #adding new column which is the difference betwheen the OPEN and LOW
            df["OPEN"] = df["OPEN"].shift(-1)
            label = df["DIFFERENCE"]
            df.drop("DIFFERENCE",axis=1,inplace=True)
            return df, label
        else:
            df["DIFFERENCE"] =df["HIGH"] - df["OPEN"]  #adding new column which is the difference betwheen the HIGH and OPEN
            df["OPEN"] = df["OPEN"].shift(-1)
            label = df["DIFFERENCE"]
            df = df.drop("DIFFERENCE",axis=1)
            return df, label
        
    
    def split(self,data, n_steps, train_Length, label):
        test_length = 50
        inputdf = data
        #making our data to numpy array
        train_x = data_frame_to_numpy(  inputdf[-(int(n_steps)+train_Length + test_length):-test_length]  )  
        train_y = (label[-(train_Length + test_length):-test_length]).to_numpy()
        test_x = data_frame_to_numpy(inputdf[-(test_length + int(n_steps)):])
        test_y = label[-(test_length):].to_numpy()
        #makeing our final predictions
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(train_Length):
           x_train.append(train_x[i:(int(n_steps) + i)])
           y_train.append(train_y[i])
        for i in range(test_length):
           x_test.append(test_x[i:(int(n_steps) + i)])
           y_test.append(test_y[i])
        x_train = np.log(x_train).reshape(train_Length,int(n_steps)*5)
        y_train = train_y
        x_test = np.log(x_test).reshape(test_length,int(n_steps)*5)
        y_test = test_y
        return np.array(x_train), np.array(y_train), np.array(x_test),np.array(y_test)
    def train(self, train_x,train_y,params):
        
        self.model = xgb.XGBRegressor(objective='reg:squarederror', params = params )
        self.model.fit(train_x, train_y)
       
        return self.model
    
    def test(self,x_test,y_test):
        self.test_prediction = self.model.predict(x_test)
        return self.test_prediction
    
    def get_model(self):
        self.data,self.label = self.labeling(self.label_name,self.train_df)
        X,Y,x,y = self.split(self.data,self.n_steps,self.train_length,self.label)
        model = self.train(X,Y,self.params)
        return model
           
    
            



            
