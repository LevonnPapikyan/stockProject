# stockProject
this project has been done by Levon Papikyan 
![plot](C:/Users/User/Desktop/me/photo_2022-06-09_15-03-47.png)

# about trading

Trading is the buying and selling of securities, such as stocks, bonds, currencies and commodities, as opposed to investing, which suggests a buy-and-hold strategy. Trading success depends on a trader's ability to be profitable over time. I tried to use deep laerning and machine learning algorithms to guide traders do their daily investments.

# Strategy for preprocessing

Firstly I want to mention that (of course it is my own opinion) the importance of understanding this code is to have a good background of mathematics. it is important to realise what kind of hyperparametrs to choose and so on. code is not recommended to use for trading.

There are many ways to scale your data or to prepare for training. So because the aim was to do daily trading I decided to include 
"open" value for the prediction of current day. Which helped to have better results compared with a models that I have trained 
without including "open". I also decided to scale data by counting logarithm of it, or just divide by numbers. I had also tried to do sigma and min-max scalling which results didn't differ much then the mothods written above.
I used three types of models for prediction. The idea was to get three independent models, and for that I wrote a class which divides data into different parts. And by the help of that class you can train models in different parts of data. 

# models

LSTM

You can try to train your own model. I used the model written in model.py and by the help of weights changing depended on the 
loss of test data (mse in my case) I continued training. You can also add the range of training models which will help you to 
get better results.

XGB

I have trained XGB model by the help of Wandb.ai which is a good platform for vizualisation.(You can also use wandb while training LSTM) There is an opportunity to train you model and at the same time do hyperparametr tunning. It helps to get better results. Not only XGB parametrs but also number of steps, train length and other values too, you can use in your sweep configuration.
 
# how to run the code

git clone https://github.com/LevonnPapikyan/stockProject  # clone

pip install -r requirements.txt  # install

 training : in model.py you can change the model artchitachure and run the weights_change.py with the stock name you want and the model will be saved in this folder. 
 
 predictions : if you have trained your wanted stock you need to save the scaling data (how you scaled your data) in csv file then run the pred_with_csv file.

 pred_with_csv file will give you excel file filled with validation data.



# Theese links may be good for to get dive in code better.

Wandb.ai   https://docs.wandb.ai/guides/sweeps

LSTM       https://colah.github.io/posts/2015-08-Understanding-LSTMs/

XGB        https://arxiv.org/pdf/1603.02754.pdf

WAVENET    https://medium.com/analytics-vidhya/wavenet-variations-for-financial-time-series-prediction-the-simple-the-directional-relu-and-the-4860d8a97af1
