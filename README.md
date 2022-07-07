# stockProject

this project has been done by Levon Papikyan 
the aim of the project is to get results which will help to do daily stock trading. I don't suggest to use the code for daily stock trading. 

There are many ways to scale your data or to preapre for training. So because the aim was to do daily trading I decided to include 
"open" value for the prediction of current day. Which helped to have better results compared with a models that I have trained 
without including "open". I also decided to scale data by counting logarithm of it, or just divide by numbers. I had also tried to do sigma and min-max scalling which results didn't differ much then the mothods written above.
I used three types of models for prediction. The idea was to get three independent models it was hardly achived and for that I wrote a class which divdes data into different parts. And by the help of that class you can train models in different parts of data.

I used three types of models for prediction. The idea was to get three independent models it was hardly achived and for that I wrote a class which divdes data into different parts. And by the help of that class you can train models in different parts of data.


LSTM

You can try to train your own model. I used the model written in model.py and the help of weights changing depended on the 
loss of test data (mse in my case) I countinued training. You can also add the range of training models which will help you to 
get better results.

XGB

I have trained XGB model by the help of Wandb.ai which is a good platform for visalisation.(You can also use wandb while training LSTM) There is an opportunity to train you model and at the same time do hyperparametr tunning. It helps to get better results. Not only XGB parametrs but also number of steps, train length and other values too you can use in your sweep configuration.