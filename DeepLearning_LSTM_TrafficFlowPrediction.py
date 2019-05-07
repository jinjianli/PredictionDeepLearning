# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:58:21 2019

@author: Jinjian LI
@objective: prediction of traffic flow with DeepLearning method based on tensorflow and keras .
@algorithm applied: LSTM
"""


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from math import sqrt
import matplotlib.pyplot as plt
import random






# split a traing Flaoting Car Data sequence into samples structures
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this samples data
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the sample data
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# input the traffic flow data from file
def ReadInputData(DataFileName):
    file_intial_data = open(DataFileName, "r")
    TrafficVolume_seq = []
    file_rows = file_intial_data.readlines()
    Num=0
    for row in file_rows:
        Num=Num+1
        row_splited = row.split()
        TrafficVolume_seq.append(int(row_splited[0]))
    return TrafficVolume_seq


# split randomly the whole data set into training and testing data set based on the given proportion 
def split_TrainingDataAndTestingDatasequence(sequenceX, sequenceY, proportion):

    TrainingDataX, TestingDataX, TrainingDataY, TestingDataY, = list(), list(), list(), list()
    for i in range(len(sequenceX)):
        if random.randint(0,1000)<1000*proportion:
            TrainingDataX.append(sequenceX[i])
            TrainingDataY.append(sequenceY[i])
        else:
            TestingDataX.append(sequenceX[i])
            TestingDataY.append(sequenceY[i])
    return array(TrainingDataX), array(TestingDataX),array(TrainingDataY), array(TestingDataY)
        
# calculation of RMSE 
def calculate_RMSE(RealValue,predictedValue):
    error=list()
    for i in range(len(RealValue)):
        error.append(pow(RealValue[i]-predictedValue[i],2))
    return sqrt(sum(error)/len(error))
    
    


#step 1 input and prepare data, set parameters 
print('step 1：input and prepare data, set parameters' )
raw_seq =ReadInputData("TrafficFlowData.txt")

# choose a number of time steps
n_steps = 16
# split into samples
X, y = split_sequence(raw_seq, n_steps)

TrainingDataX, TestingDataX, TrainingDataY, TestingDataY=split_TrainingDataAndTestingDatasequence(X,y, 0.8)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1

TrainingDataX = TrainingDataX.reshape((TrainingDataX.shape[0], TrainingDataX.shape[1], n_features))








# step 2 define and fit model
print ('step 2: define and fit model')
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(TrainingDataX, TrainingDataY, epochs=200, verbose=0)

# step 3 demonstrate prediction performance 
print('step 3: demonstrate prediction performance ')
x_input=TestingDataX
x_input = x_input.reshape((x_input.shape[0], n_steps, n_features))
predictedValue = model.predict(x_input, verbose=0)
erro=calculate_RMSE(TestingDataY,predictedValue)
print('prediction MSME for testing data set: ')
print(erro)
X = X.reshape((X.shape[0], n_steps, n_features))
predictedValue = model.predict(X, verbose=0)
plt.figure()
plt.xlabel('Time (h)')
plt.ylabel('Traffic flow (Vehs/h)')
plt.plot(y,color='blue', label='Real traffic flows')
plt.plot(predictedValue, color='red', label='Predicted traffic flows')
plt.show
plt.savefig("comparison between real and predicted traffic flows.svg")
# =============================================================================

