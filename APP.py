import keras
from sklearn.neural_network import MLPRegressor
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import keras
import pandas as pd
import numpy as np
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time 
import quandl
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
quandl.ApiConfig.api_key = "p_aAK6D87z8PVVvX4kYM"
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib import style
from sklearn.linear_model import LinearRegression
import quandl
import datetime
style.use('ggplot')
from sklearn.metrics import r2_score

#########
print('IMPORTING DATA FOR FORECASTING')
dataset = pd.read_csv('Churn_Modelling.csv')
churn_X = dataset.iloc[:, 3:13].values
churn_y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
churn_X[:, 1] = labelencoder_X_1.fit_transform(churn_X[:, 1])
labelencoder_X_2 = LabelEncoder()
churn_X[:, 2] = labelencoder_X_2.fit_transform(churn_X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
churn_X = onehotencoder.fit_transform(churn_X).toarray()
churn_X = churn_X[:, 1:]
churn_X_train,churn_X_test, churn_y_train, churn_y_test = train_test_split(churn_X, churn_y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
churn_X_train = sc.fit_transform(churn_X_train)
churn_X_test = sc.transform(churn_X_test)


mini= MinMaxScaler(feature_range=(0,1)) 
# Importing the dataset
credit= pd.read_csv('Credit_Card_Applications.csv')
credit = credit.drop('CustomerID',axis=1)
X = credit.iloc[:, :-1].values
y = credit.iloc[:, -1].values




###############
AAPL = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['close','low','high','open','volume','ex-dividend',
                                                             'split_ratio','adj_open','adj_low',
                                                             'adj_volume','adj_close'] }, 
                        ticker = ['AAPL'], 
                        date = { 'gte': '1980-12-31', 'lte': '2019-5-28' })
# training_AAPL = pd.DataFrame(AAPL)
AAPL_df = AAPL.reset_index()
AAPL_prices = AAPL_df['high'].tolist()
AAPL_dates = AAPL_df.index.tolist()
#Convert to 1d Vector
AAPL_dates = np.reshape(AAPL_dates, (len(AAPL_dates), 1))
AAPL_prices = np.reshape(AAPL_prices, (len(AAPL_prices), 1))

###########
MSFT = quandl.get_table('WIKI/PRICES', qopts = { 'columns': [ 'close','low','high','open','volume','ex-dividend',
                                                             'split_ratio','adj_open','adj_low',
                                                             'adj_volume','adj_close'] },
                        ticker = ['MSFT'], 
                        date = { 'gte': '1986-03-31', 'lte': '2019-5-28' })
# training_MSFT= pd.DataFrame(MSFT)
MSFT_df = MSFT.reset_index()
MSFT_prices = MSFT_df['high'].tolist()
MSFT_dates = MSFT_df.index.tolist()

#Convert to 1d Vector
MSFT_dates = np.reshape(MSFT_dates, (len(MSFT_dates), 1))
MSFT_prices = np.reshape(MSFT_prices, (len(MSFT_prices), 1))
##################
ALK=quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['close','low','high','open','volume','ex-dividend',
                                                          'split_ratio','adj_open','adj_low',
                                                          'adj_volume','adj_close'] },
                     ticker = ['ALK'],
                     date = { 'gte': '1980-03-31', 'lte': '2019-5-28' })
# training_ALK = pd.DataFrame(ALK)
ALK_df = ALK.reset_index()
ALK_prices = ALK_df['high'].tolist()
ALK_dates = ALK_df.index.tolist()
 
#Convert to 1d Vector
ALK_dates = np.reshape(ALK_dates, (len(ALK_dates), 1))
ALK_prices = np.reshape(ALK_prices, (len(ALK_prices), 1))
 
###########

AAWW =quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['close','low','high','open','volume','ex-dividend',
                                                            'split_ratio','adj_open','adj_low',
                                                            'adj_volume','adj_close'] }, 
                       ticker = ['AAWW'], 
                       date = { 'gte': '2004-07-30', 'lte': '2019-5-28' })
# training_AAWW = pd.DataFrame(AAWW)
AAWW_df = AAWW.reset_index()
AAWW_prices = AAWW_df['high'].tolist()
AAWW_dates = AAWW_df.index.tolist()
 
#Convert to 1d Vector
AAWW_dates = np.reshape(AAWW_dates, (len(AAWW_dates), 1))
AAWW_prices = np.reshape(AAWW_prices, (len(AAWW_prices), 1))


def credit_fraud(X):
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
# TODO: Maximum price of the data
    maximum_price = np.max(X)
    minimum_price = np.min(X)
    # TODO: Mean price of the data
    mean_price = np.mean(X)
    # TODO: Median price of the data
    median_price = np.median(X)
    # TODO: Standard deviation of prices of the data
    std_price = np.std(X)
    # Show the calculated statistics
    print( "Conducting Analysis On The Data Given:\n")
    print('min price: {}'.format(minimum_price))
    print('max price: {}'.format(maximum_price))
    print('mean price: {}'.format(mean_price))
    print( 'median of prices: {}'.format(median_price))
    print("Standard deviation of prices: {} ".format(std_price))
    # Training the SOM
    from minisom import MiniSom
    som = MiniSom(x = 10, y = 10, input_len =14, sigma = 1.0, learning_rate = 0.5) #Quanization = 1
    som.random_weights_init(X)
    som.train_random(data = X, num_iteration = 300)
#     som.quantization(X)
    # Visualizing the results
    from pylab import bone, pcolor, colorbar, plot, show
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        print(w)
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor = colors[y[i]],
             markerfacecolor = 'b',
             markersize = 10,
             markeredgewidth = 2)
    plt.show()
    # Finding the frauds
    mappings = som.win_map(X)
    display(mappings)
    frauds = np.concatenate((mappings[(8,1)], mappings[(8,1)]), axis = 0)
    if frauds in w: 
        print('fraud detected')
    else:
        print('Fraud Not Detected')
    return som,frauds


def credit_customer():
    A1 = [int(float(input('Please enter your A1:')))]
    A2 = [int(float(input('Please enter your A2:')))]
    A3 = [int(float(input('Please enter your A3:')))]
    A4 = [int(float(input('Please enter your A4:')))]
    A5 = [int(float(input('Please enter your A5:')))]
    A6 = [int(float(input('Please enter your A6:')))]
    A7 = [int(float(input('Please enter your A7:')))]
    A8 = [int(float(input('Please enter your A8:')))]
    A9 = [int(float(input('Please enter your A9:')))]
    A10 = [int(float(input('Please enter your A10:')))]
    A11 = [int(float(input('Please enter your A11:')))]
    A12 = [int(float(input('Please enter your A12:')))]
    A13 = [int(float(input('Please enter your A13')))]
    A14 = [int(float(input('Please enter your A14')))]
    Class = [int(float(input('Please enter your Credit Class ')))]
    client_data = [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,Class] #Customer ID
    for i, fraud in enumerate(credit_fraud(client_data)):
        print(i+1, fraud) 
    return client_data
    
def stock():
    AAPL = 'AAPL'
    AAWW = 'AAWW'
    ALK = 'ALK'
    MSFT = 'MSFT'
    stock = input('Please Enter the stock you would like to predict {} {} {} {}'.format(AAPL,AAWW,ALK,MSFT)) 
    
    while True:
        if stock == AAPL:
            model.fit(AAPL_dates,AAPL_prices,batch_size=batch_size,epochs=10,validation_split=.67)#callbacks=[tensorboard])
            X_train,X_test,y_train,y_test = train_test_split(AAPL_dates,AAPL_prices,test_size=.67)
            regressor = LinearRegression(normalize=True,n_jobs=-1)
            regressor.fit(X_train,y_train)
            print('transforming the ADABOOSTREGRESSOR DATA')
            print('finished')
            print(X_test.shape)
            print(AAPL_dates)
            print('Predicted Close Stock Price is: {}'.format(regressor.predict(X_test[4693:4694])))
            
            plt.scatter(X_train, y_train, color='yellow', label= 'Actual Price') #plotting the initial datapoints
            plt.plot(X_test, regressor.predict(X_test), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
            plt.title('Linear Regression | Time vs. Price')
            plt.legend()
            plt.xlabel('Date Integer')
            plt.show()
            print(r2_score(AAPL_prices,regressor.predict(AAPL_dates)))
            forecast_out = int(math.ceil(0.01 * len(AAPL)))
            AAPL_lately = AAPL_prices[-forecast_out:]
            forecast_set = regressor.predict(AAPL_lately)
            if forecast_set >= AAPL_lately:
                print('I Suggest You Sell A Share')
            else:
                print('I Suggest You Buy a  Share')
            break 
            continue
            
        elif stock == AAWW:
            X_train,X_test,y_train,y_test = train_test_split(AAWW_dates,AAWW_prices,test_size=.65)
            regressor = LinearRegression(normalize=True,n_jobs=-1)      
            regressor.fit(X_train,y_train)
            regressor= AdaBoostRegressor((regressor), n_estimators = 10, learning_rate = .4,loss='linear')
            regressor.fit(X_train,y_train.ravel())
            print(X_test.shape)
            print(AAWW_dates)
            print('Predicted High  Stock Price is: {}'.format(regressor.predict(X_test[2024:2025])))
            
            plt.scatter(X_train, y_train, color='yellow', label= 'Actual Price') #plotting the initial datapoints
            plt.plot(X_test, regressor.predict(X_test), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
            plt.title('Linear Regression | Time vs. Price')
            plt.legend()
            plt.xlabel('Date Integer')
            plt.show()
            print(r2_score(AAWW_prices,regressor.predict(AAWW_dates)))    
            forecast_out = int(math.ceil(0.01 * len(AAWW)))
            AAWW_lately = AAWW_prices[-forecast_out:]
            forecast_set = regressor.predict(AAWW_lately)
            if forecast_set >= AAWW_lately:
                print('I Suggest You Sell A Share')
            else:
                print('I Suggest You Buy a  Share')
            break 
            continue
            
        elif stock == MSFT:
            model.fit(MSFT_dates,MSFT_prices,batch_size=batch_size,epochs=100,validation_split=.45)#callbacks=[tensorboard])
            X_train,X_test,y_train,y_test = train_test_split(MSFT_dates,MSFT_prices,test_size=.67)
            regressor = LinearRegression(normalize=True,n_jobs=-1)
            regressor.fit(X_train,y_train)#callbacks=[tensorboard]))
            regressor= AdaBoostRegressor((regressor), n_estimators = 10, learning_rate = .4,loss='linear')
            regressor.fit(X_train,y_train.ravel())
            print(X_test.shape)
            print(MSFT_dates)
            print('Predicted High Stock Price is: {}'.format(regressor.predict(X_test[5403:5404])))
            plt.scatter(X_train, y_train, color='yellow', label= 'Actual Price') #plotting the initial datapoints 
            plt.plot(X_test,regressor.predict(X_test), color='red', linewidth=3, label = 'Predicted Price') #regressor.predict, #plotting the line made by linear regression
            plt.title('Linear Regression | Time vs. Price')
            plt.legend()
            plt.xlabel('Date Integer')
            plt.show()
            print(r2_score(MSFT_prices,regressor.predict(MSFT_dates)))        
            forecast_out = int(math.ceil(0.01 * len(MSFT)))
            MSFT_lately = MSFT_prices[-forecast_out:] 
            forecast_set = regressor.predict(MSFT_lately)
            if forecast_set >= MSFT_lately:
                print('I Suggest You SELL A Share')
            else:
                print('I Suggest You BUY A SHARE')
            break 
            continue
            
        elif stock ==ALK:
            model.fit(ALK_dates,ALK_prices,batch_size=batch_size,epochs=100,validation_split=.67)#callbacks=[tensorboard]))
            X_train,X_test,y_train,y_test = train_test_split(ALK_dates,ALK_prices,test_size=.6)
            regressor = LinearRegression(normalize=True,n_jobs=-1)
            regressor.fit(X_train,y_train) #X=dates y = prices
            regressor= AdaBoostRegressor((regressor), n_estimators = 10, learning_rate = .4,loss='linear')
            print(regressor)
            regressor.fit(X_train,y_train.ravel())

            
            print(ALK_dates)
            print('Predicted High Stock Price is: {}'.format(regressor.predict(X_test[5329:5330])))
            plt.scatter(X_train, y_train, color='yellow', label= 'Actual Price') #plotting the initial datapoints 
            plt.plot(X_test,regressor.predict(X_test), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
            plt.title('Linear Regression | Time vs. Price')
            plt.legend()
            plt.xlabel('Date Integer')
            plt.show()
            print(r2_score(ALK_prices,regressor.predict(ALK_dates))) 
            forecast_out = int(math.ceil(0.01 * len(ALK)))
            ALK_lately = ALK_prices[-forecast_out:]
            forecast_set = regressor.predict(ALK_lately)
            if forecast_set >= ALK_lately:
                print('I Suggest You Sell A Share')
            else:
                print('I Suggest You BUY A SHARE')
            
            break
            continue
        else:
            stock2 = input('Sorry {}, \nThis Operation is invalid, please choose from\n {},\n{},\n{},\n or {}.'.format(name,AAPL,ALK,AAWW,MSFT))
    return stock

def churn_customer():
    batch_size = 710
    dropout = 0.2
    visible = Input(shape=(11,))
    hidden1 = Dense(105, activation='relu')(visible)
    hidden1 = Dense(50, activation='relu')(hidden1)
    hidden1 = Dense(50, activation='relu')(hidden1)
    hidden1 = Dense(50, activation='relu')(hidden1)
    hidden1 = Dense(50, activation='relu')(hidden1)
    hidden1 = Dense(50, activation='relu')(hidden1)
    hidden2 = Dense(50, activation='relu')(hidden1)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden2 = Dense(50, activation='relu')(hidden2)
    hidden3 = Dense(50, activation='relu')(hidden2)
#             hidden3 = Dense(50, activation='relu')(hidden3)
#             hidden3 = Dense(50, activation='relu')(hidden3)
#             hidden3 = Dense(50, activation='relu')(hidden3)
    output = Dense(1, activation='relu')(hidden3) 
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['categorical_accuracy'])
    model.summary()
    #         tensorboard = TensorBoard(log_dir="logs/{} ".format(NAME))
    model.fit(churn_X_train,churn_y_train,batch_size=batch_size,epochs=100,validation_split=.67)#callbacks=[tensorboard]))
    y_pred = model.predict(churn_X_test)
    y_pred = (y_pred > 0.5)
    
    new_prediction = model.predict(sc.transform(np.array([[int(float(input('Please Enter your Geography'))),
                                                           int(input('Please Enter your Credit Score')), int(input('Please Enter your 0 for female or 1 for male')), int(input('Please Enter your AGE')),
                                                          int(input('Please Enter your tenure ')), int(input('Please Enter your Balance')), int(input('Please Enter your Num Of Products')), int(input('Please Enter 0 if you do not have a credit card Enter 1 if you do')), int(input('Please Enter 0 If not an active member Enter 1 if you are')),
                                                           int(input('Please Enter your Estimated Salary')), int(input('Have you Ever left this company Enter 0 for no enter 1 for yes'))]])))
    
    new_prediction = (new_prediction > 0.5) 
    print('It Is Predicted {} That the Customer Will Return To Bank'.format(new_prediction))
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(churn_y_test, y_pred)
    print(cm)
 
    return cm,y_pred,model

r = 'Risk Assessment'
cm = 'Churn Modeling'
cr = 'Credit Card Fraud Detection'
s = 'Stock Prediction'

name = input("Hello, I Am Roger\_(^^)_/, I Hope You Are Doing Well, What Is Your Name? \n")
print('Hello, ',name )
stock_risk = input('How Can I help you today {}.\n Please Pick From One of The Folopening\n{},\n{},\n{},\nor {}. \n '.format(name,s,r,cm,cr))

while True:
    if stock_risk == s:
        print('Okay Stock Prediction')
        print('Welcome To My Stock Prediction Application')
        print('Choose A Stock You Would Like to Predict')
        stock()
        print('Exiting {} Application, Thank you, {}'.format(stock_risk,name))
        break 
        continue
    elif stock_risk == r:
        print('Okay Risk Assessment') 
        print('Welcome To My Risk Assessment Portfolio Application')
        print('Under Construction')
        print('Exiting {} Application, Thank you, {}'.format(stock_risk,name))
        break
        continue
    elif stock_risk == cm:
        print('Okay Churn Modeling')
        print('Welcome To My Churn Modeling Application')
        churn_customer()
        print('Exiting {} Application, Thank you, {}'.format(stock_risk,name))
        break 
        continue 
    elif stock_risk == cr:
        print('Okay Credit Card Fraud Detection') 
        print('Welcome To My Credit Card Fraud Application')
        credit_customer()
        print('Exiting {} Application,and Thank you, {}'.format(stock_risk,name))
        break
        continue
    else:
        stock_risk2 = input('Sorry {}, \nThis Operation is invalid, please choose from\n {},\n{},\n{},\n or {}.'.format(name,s,r,cm,cr))
        if stock_risk2 == s:
            print('Thank You {},\n Lets Contine Now With The {} AppLication \n'.format(name,stock_risk2))
            print('Welcome To My Stock Prediction Application')
            print('Choose A Stock You Would Like to Predict')
            stock()
            print('Exiting {} Application, Thank you, {}'.format(stock_risk2,name))
            break
            continue
        elif stock_risk2 == r:
            print('Thank You {},\n Lets Contine Now With The {} Application \n'.format(name,stock_risk2))
            print('Welcome To My Risk Assessment Portfolio Application')
            print('Under Construction')
            print('Exiting {} Application, Thank you, {}'.format(stock_risk2,name))
            break 
            continue 
        elif stock_risk2 == cm:
            print('Thank You {},\n Lets Contine Now With The {} Application \n'.format(name,stock_risk2))
            print('Welcome To My Churn Modeling Application')
            churn_customer()
            print('Exiting {} Application, Thank you, {}'.format(stock_risk2,name))
            break 
            continue
        elif stock_risk2 == cr:
            print('Thank You {},\n Lets Contine Now With The {} Application \n'.format(name,stock_risk2))
            print('Welcome To My Credit Card Fraud Application')
            credit_customer()
            break 
            continue
            print('Exiting {} Application, Thank you, {}'.format(stock_risk2,name))