import numpy as np
import pandas as pd
import random
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, GRU, Flatten
from keras.models import Sequential  
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import models 
import seaborn as sns

# %%
# from google.colab import drive
# drive.mount("/content/gdrive")

# %% [markdown]
# Read different columns and delete zeros from them

# %%
data_train = pd.read_csv('training_dataset.csv')
data1 = np.array(data_train['firewall_rx_bytes']).reshape(-1,1)
data2 = np.array(data_train['dpi_rx_bytes']).reshape(-1,1)
data3 = np.array(data_train['ids_rx_bytes']).reshape(-1,1)
data4 = np.array(data_train['lb_rx_bytes']).reshape(-1,1)

# %% [markdown]
# Normalizatiion

# %%
scaler = MinMaxScaler(feature_range=(0,20)) 
data_scaled1 = scaler.fit_transform(data1)
data_scaled2 = scaler.fit_transform(data2)
data_scaled3 = scaler.fit_transform(data3)
data_scaled4 = scaler.fit_transform(data4)

# %% [markdown]
# 
# Split train data and test data

# %%
train_size1 = int(len(data1)*0.8)
train1 = data_scaled1[0:train_size1,:]
test1 = data_scaled1[train_size1:,:]

train_size2 = int(len(data2)*0.8)
train2 = data_scaled2[0:train_size2,:]
test2 = data_scaled2[train_size2:,:]

train_size3 = int(len(data3)*0.8)
train3 = data_scaled3[0:train_size3,:]
test3 = data_scaled3[train_size3:,:]

train_size4 = int(len(data4)*0.8)
train4 = data_scaled4[0:train_size4,:]
test4 = data_scaled4[train_size4:,:]

# %% [markdown]
# Preproccessing data for LSTM models

# %%
# Create input dataset
def create_dataset (X, look_back = 1):
    Xs, ys = [], []
 
    for i in range(len(X)-look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        ys.append(X[i+look_back])
 
    return np.array(Xs), np.array(ys)

# %%
LOOK_BACK = 4
X1_train, y1_train = create_dataset(train1,LOOK_BACK)
X1_test, y1_test = create_dataset(test1,LOOK_BACK)

X2_train, y2_train = create_dataset(train2, LOOK_BACK)
X2_test, y2_test = create_dataset(test2, LOOK_BACK)

X3_train, y3_train = create_dataset(train3,LOOK_BACK)
X3_test, y3_test = create_dataset(test3,LOOK_BACK)

X4_train, y4_train = create_dataset(train4,LOOK_BACK)
X4_test, y4_test = create_dataset(test4,LOOK_BACK)

# %% [markdown]
# Create models(PMs)

# %%
def model1(): # FW
    model = Sequential()
    model.add(GRU(48, return_sequences=True, input_shape=(4,1)))
    model.add(GRU(32, activation='tanh',return_sequences=True))
    model.add(GRU(32, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005759),loss='mse', metrics='mae')
    return model

# %%
def model2(): # DPI
    model = Sequential()
    model.add(GRU(64,return_sequences=True, input_shape=(4,1)))
    model.add(GRU(32, activation='tanh', return_sequences=True))
    model.add(GRU(16, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0006384),loss='mse', metrics='mae')
    return model

# %%
def model3(): # IDS
  model = Sequential()
  # Input layer
  model.add(LSTM(32,return_sequences=True, input_shape=(4,1)))
  # Hidden layer
  model.add(LSTM(48, activation='tanh'))
  # Output layer
  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=0.000616),loss='mse', metrics='mae')
  return model

# %%
def model4(): # LB
    model = Sequential()
    model.add(GRU(48,return_sequences=True, input_shape=(4,1)))
    model.add(GRU(48, activation='tanh', return_sequences=True))
    model.add(GRU(48, activation='tanh', return_sequences=True))
    model.add(GRU(16, activation='tanh', return_sequences=True))
    model.add(GRU(16, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.000871),loss='mse', metrics='mae')
    return model

# %%
def model_SFC(): # SFC  # The structure by keras tuner
  model = Sequential()
  # Input layer
  model.add(LSTM(32,return_sequences=True, input_shape=(4,1)))
  # Hidden layer
  model.add(LSTM(64,return_sequences=True, activation='tanh'))
  model.add(LSTM(48,return_sequences=True, activation='tanh'))
  model.add(LSTM(48,return_sequences=True, activation='tanh'))
  model.add(LSTM(16,activation='tanh'))
  # Output layer
  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=0.00021),loss='mse', metrics='mae')
  return model

# %%
def model_SFC_GRU(): # SFC  # The structure by keras tuner
  model = Sequential()
  # Input layer
  model.add(GRU(16,return_sequences=True, input_shape=(4,1)))
  
  model.add(GRU(48,return_sequences=True, activation='tanh'))
  model.add(GRU(48, activation='tanh'))
  # Output layer
  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=0.0005),loss='mse', metrics='mae')
  return model

# %%
model1 = model1()
model2 = model2()
model3 = model3()
model4 = model4()

# %%
lstm_SFC = model_SFC()
gru_SFC = model_SFC_GRU()

# %% [markdown]
# Fit models

# %%
def fit_model(model,X_train,y_train):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)
    history = model.fit(X_train, y_train, epochs = 5,  validation_split = 0.2,batch_size = 16, shuffle = False, callbacks = [early_stop], verbose=0)
    return model

# %%
# sub1 = fit_model(model1,X1_train,y1_train)
# model1.save("./model1.tf")

# %%
# sub2 = fit_model(model2,X2_train,y2_train)
# model2.save("./model2.tf")

# %%
# sub3 = fit_model(model3,X3_train,y3_train)
# model3.save("./model3.tf")

# %%
# sub4 = fit_model(model4,X4_train,y4_train)
# model4.save("./model4.tf")

# %%
sub1 = models.load_model("model1.tf")
sub2 = models.load_model("model2.tf")
sub3 = models.load_model("model3.tf")
sub4 = models.load_model("model4.tf")

# %% [markdown]
# Create SFC dataset

# %%
# FW, DPI, IDS, LB
start1 = 0
start2 = 0
start3 = 0
start4 = 0
SFC = []
while(start1 <= len(train1) or start2 <= len(train2) or start3 <= len(train3) or start4 <= len(train4)): # until the end of datasets(It should be change later)
    d = random.randint(4,10)
    FW = train1[start1:start1+d]
    start1 = start1+d
    SFC.append(FW)

    d = random.randint(4,10)
    DPI = train2[start2:start2+d]
    start2 = start2+d
    SFC.append(DPI)

    d = random.randint(4,10)
    IDS = train3[start3:start3+d]
    start3 = start3+d
    SFC.append(IDS) 

    d = random.randint(4,10)
    LB = train4[start4:start4+d]
    start4 = start4+d
    SFC.append(LB)

SFC_train = np.concatenate(SFC)
#SFC = np.concatenate(SFC).reshape(1,-1)    

# %% [markdown]
# RL

# %%
from gym import Env
from gym.spaces import  Discrete, Box
import numpy as np
import time
import math
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from collections import deque

# %% [markdown]
# Hyper Parameters

# %%
gamma = 0.95 # Discount factor
alpha = 0.001 # Learning rate
epsilon = 1
epsilon_min = 0.0000001
epsilon_decay = 0.998
batch_size = 128
episodes = 15

# %% [markdown]
# DQN class

# %%
from tabnanny import verbose


class TrafficPrediction:
  def __init__(self,alpha,gamma,epsilon,epsilon_min,epsilon_decay):
    self.action_space = Discrete(4)
    self.nS = 4
    self.nA = self.action_space.n
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.gamma = gamma
    self.alpha = alpha
    self.loss = []
    self.model = self.build_model()
    self.memory = deque([], maxlen=2500)

  def build_model(self):
    model = Sequential()
    model.add(Dense(24, input_dim=self.nS ,activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.nA, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self.alpha))
    return model

  def action(self, state): #Find action(predictor)
    if np.random.rand() <= self.epsilon: #Explore
        return random.randrange(self.nA)
    action_vals = self.model.predict(state, verbose=0) #Exploit: Use the NN to predict the correct action from this state  
    #print("action selected by model \n")
    return np.argmax(action_vals[0])

  def test_action(self, state): # Exploit (find suitable predictor for test)
    action_vals = self.model.predict(state, verbose=0)
    return np.argmax(action_vals[0])

  def store(self, state, action, reward, nstate):
    # state: seq4, action:single value, reward:single value, nstate:seq4 => Totally store 10 integers values
    # Store observations
    self.memory.append((state,action,reward,nstate))

  def get_reward(self,mse):
    if(mse>=0 and mse<0.1):
        reward = +5
    elif(mse>=0.1 and mse<0.2):
        reward = +4
    elif(mse>=0.2 and mse<0.3):
        reward = +3
    elif(mse>=0.3 and mse<0.4):
        reward = +2
    elif(mse>=0.4 and mse<0.5):
        reward = +1
    elif(mse>=0.5 and mse<0.6):
        reward = 0
    elif(mse>=0.6 and mse<0.7):
        reward = -1
    elif(mse>=0.7 and mse<0.8):
        reward = -2
    elif(mse>=0.8 and mse<0.9):
        reward = -3
    elif(mse>=0.9):
        reward = -4                                
    return reward

  def step(self,action,next_value): # choose a predictor and evaluates its accuracy
    #MSEs = []
    next_value = [next_value]
    if(action == 0):      
      next_predicted_value = sub1.predict(state, verbose=0)
      mse0 = mean_squared_error(next_value,next_predicted_value.flatten())
      reward = self.get_reward(mse0)
      #MSEs.append(mse0)
    elif(action == 1):    
      next_predicted_value = sub2.predict(state, verbose=0)
      mse1 = mean_squared_error(next_value,next_predicted_value.flatten())
      reward = self.get_reward(mse1)
      #MSEs.append(mse1)
    elif(action == 2):
      next_predicted_value = sub3.predict(state, verbose=0)
      mse2 = mean_squared_error(next_value,next_predicted_value.flatten())
      reward = self.get_reward(mse2)
      #MSEs.append(mse2)
    elif(action == 3):
      next_predicted_value = sub4.predict(state, verbose=0)
      mse3 = mean_squared_error(next_value,next_predicted_value.flatten())
      reward = self.get_reward(mse3)
      #MSEs.append(mse3)

    #print("state: ",state)
    nstate = np.array([None]*4).reshape(1,-1) 
    nstate[0,0:3] = state[0,1:4]
    nstate[0,3] = next_value.pop()
    nstate = nstate.astype(float)
    return nstate,reward,next_predicted_value

  def reset(self):
    state = random.sample(range(700,10000),4)
    total_reward = 0
    return self.state

  def expreince_replay(self, batch_size, time): 
    #Execute the expreince replay
    #print("now training is based on experience: ")
    minibatch = random.sample( self.memory, batch_size ) # Randomly sample from memory 
    #print(len(minibatch))    
    x, y = [], []
    np_array = np.array(minibatch)
    st = np.zeros((0,self.nS)) # states
    nst = np.zeros((0,self.nS)) # nStates

    for i in range(len(np_array)): #Creating the state and next state np_array (state is the sequence of the last 4 interval values and next state is the fifth value)
      st = np.append(st, np_array[i,0])  
      nst = np.append(nst, np_array[i,3]) 

    st = np.reshape(st,(batch_size,self.nS))
    nst = np.reshape(nst,(batch_size,self.nS))
    st_predict = self.model.predict(st, verbose=0) 
    nst_predict = self.model.predict(nst, verbose=0)   
    index = 0
    for state, action, reward, nstate in minibatch:
      x.append(state)
      # Predict from state
      nst_action_predict_model = nst_predict[index]
      target = reward + self.gamma * np.argmax(nst_action_predict_model)
      target_f = st_predict[index]
      target_f[action] = target
      y.append(target_f)
      index += 1
    #Reshape for keras Fit
    x_reshape = np.array(x).reshape(batch_size, 4)
    y_reshape = np.array(y)
    epoch_count = 1 #Epochs is the number or iterations
    hist = self.model.fit(x_reshape,y_reshape,epochs=epoch_count,verbose=0)
    #Graph Losses
    #print("history is",hist.history)
    for i in range(epoch_count):
      self.loss.append( hist.history['loss'][i])
    #Decay epsilon
    if self.epsilon > self.epsilon_min and time % 3000 == 0:
      self.epsilon *= self.epsilon_decay
      counter = 0

# %% [markdown]
# Creat the agent

# %%
dqn = TrafficPrediction(alpha,gamma,epsilon,epsilon_min,epsilon_decay)

# %% [markdown]
# # Train the agent

# %%
import pickle
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
total_rewards_train = [] # Store rewards for graphing
rew = []
predicted_workloads_train = []
epsilons = [] # Store the Explore/Exploit
TEST_Episodes = 0
#test_data = SFC
#test = train_scaled_rl

for e in range(episodes):
  episode_predicted_workloads_train = []
  l = 0
  tot_rewards = 0
  counter = 0
  state = np.array(SFC_train[0:4]).reshape(1,-1)
  #state = np.reshape(state, [1,4])
  for time in range(int(len(SFC_train)/100)-5): #len(test_data1)-5   for time in range(int(len(SFC_train)/100)-5)
    action = dqn.action(state)
    nstate,reward,next_predicted_value = dqn.step(action,SFC_train[l+5])
    episode_predicted_workloads_train.append(next_predicted_value)
    #print("Episode: {}, Interval: {}, state:{}, action:{}, reward:{}, nstate:{}, next_predicted_value:{} \n" .format(e,time,state,action,reward,nstate,next_predicted_value))
    l += 1
    rew.append(reward)
    tot_rewards += reward
    dqn.store(state,action,reward,nstate)
    #print("length of memory: ",len(dqn.memory))
    state = nstate
    if (time==int(len(SFC_train)/100)-6): #len(test_data)-6)  if (time==int(len(SFC_train)/100)-6)
      total_rewards_train.append(tot_rewards)
      epsilons.append(dqn.epsilon)
      print("Episode: {},interval: {}/{}, score: {}, epsilon: {}"
            .format(e, time,len(SFC_train), tot_rewards, dqn.epsilon))
      break
    if len(dqn.memory) > batch_size:
      dqn.expreince_replay(batch_size,time)

  # if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
  #     #Set the rest of the EPISODES for testing
  #     TEST_Episodes = EPISODES - e 
  #     TRAIN_END = e 
  #     break
  print(f"tot_reward episod {e}: {tot_rewards}")
  predicted_workloads_train.append(episode_predicted_workloads_train)

  # Final save after all episodes
dqn.model.save('dqn_model.h5')
with open('dqn_agent.pkl', 'wb') as f:
    pickle.dump(dqn, f)
print("Final agent state saved.")

# %% [markdown]
# ## Test the agent
# 

# %% [markdown]
# Create SFC test data

# %%
# FW, DPI, IDS, LB
start1 = 0
start2 = 0
start3 = 0
start4 = 0
SFC_test = []
while(start1 <= len(test1) or start2 <= len(test2) or start3 <= len(test3) or start4 <= len(test4)): # until the end of datasets(It should be change later)
    d = random.randint(4,10)
    FW = test1[start1:start1+d]
    start1 = start1+d
    SFC_test.append(FW)

    d = random.randint(4,10)
    DPI = test2[start2:start2+d]
    start2 = start2+d
    SFC_test.append(DPI)

    d = random.randint(4,10)
    IDS = test3[start3:start3+d]
    start3 = start3+d
    SFC_test.append(IDS) 

    d = random.randint(4,10)
    LB = test4[start4:start4+d]
    start4 = start4+d
    SFC_test.append(LB)

SFC_test = np.concatenate(SFC_test)
#SFC = np.concatenate(SFC).reshape(1,-1)    

# %%
#Test the agent that was trained
#In this section we ALWAYS use exploit don't train any more
# For test now, we use data2 as test dataset
predicted_workloads_test = []
cpu_util_list = []
test_rewards = []
total_rewards_test = []
for e_test in range(1):
  episode_predicted_workloads_test = []
  state = np.array(SFC_test[0:4]).reshape(1,-1)
  #state = np.reshape(state,(1,4))
  tot_rewards = 0
  l = 0
  for t_test in range(int((len(SFC_test)/50)-5)):
      true_value = SFC_test[l+5]
      #state = np.reshape(state,(1,4)) 
      action = dqn.test_action(state) 
      #state = np.reshape(state, (1,4))
      nstate, reward, next_predicted_value = dqn.step(action,true_value) 
      episode_predicted_workloads_test.append(next_predicted_value)
      #print("Episode: {}, Interval: {}, state:{}, action:{}, reward:{}, nstate:{}, next_predicted_value:{} \n" .format(e_test,t_test,state,action,reward,nstate,next_predicted_value))
      test_rewards.append(reward)
      #cpu_util = resource_allocator(next_predicted_value) 
      #cpu_util_list.append(cpu_util) 
      tot_rewards += reward
      #DON'T STORE ANYTHING DURING TESTING
      state = nstate
      l += 1
      if t_test == int(len(SFC_test)/50)-6: 
          predicted_workloads_test.append(episode_predicted_workloads_test)
          total_rewards_test.append(tot_rewards)
          epsilons.append(0) #We are doing full exploit
          print("episode: {}/{}, score: {}, e: {}"
                .format(e_test, TEST_Episodes, tot_rewards, 0))
          break;

# Save test results
with open('test_results.pkl', 'wb') as f:
    pickle.dump({
        'predicted_workloads_test': predicted_workloads_test,
        'cpu_util_list': cpu_util_list,
        'test_rewards': test_rewards,
        'total_rewards_test': total_rewards_test
    }, f)
print("Test results saved.")

# %%
len(SFC_test)

# %%
RLPM_output = np.concatenate(predicted_workloads_test)
RLPM_output = RLPM_output.reshape(1,-1)

# %%
#sub_gru_SFC = fit_model(gru_SFC, X_train_SFC, y_train_SFC)
#gru_SFC.save("./model_SFC_GRU.tf")
SFC_GRU = models.load_model("model_SFC_GRU.tf")

# %%
#sub_lstm_SFC = fit_model(lstm_SFC,X_train_SFC,y_train_SFC)
#lstm_SFC.save("./model_SFC_LSTM.tf")
SFC_LSTM = models.load_model("model_SFC_LSTM.tf")

# %% [markdown]
# Prepare data for comparison

# %%
real_data_hourly = SFC_test[0:59].reshape(1,-1).flatten()
predicted_RL_hourly = np.concatenate(episode_predicted_workloads_test[0:59]).reshape(1,-1).flatten()

# %%
real_data = SFC_test[0:1439].reshape(1,-1).flatten()
predicted_RL = np.concatenate(episode_predicted_workloads_test[0:1439]).reshape(1,-1).flatten()

# %%
24*60


# %% [markdown]
# Evaluate LSTM model for SFC dataset

# %%
LSTM_SFC_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC = SFC_LSTM.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    LSTM_SFC_preds.append(pred_SFC)

LSTM_SFC_preds = np.concatenate(LSTM_SFC_preds).reshape(1,-1)
a = LSTM_SFC_preds
a1 = np.average(a)

# %% [markdown]
# Evaluate GRU model for SFC dataset

# %%
GRU_SFC_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC_gru = SFC_GRU.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    GRU_SFC_preds.append(pred_SFC_gru)

GRU_SFC_preds = np.concatenate(GRU_SFC_preds).reshape(1,-1)
c = GRU_SFC_preds
c1 = np.average(c)

# %%
b = np.concatenate(episode_predicted_workloads_test[0:59]).reshape(1,-1)
b1 = np.average(b)
((b1-a1)/a1)*100

# %%
((b1-c1)/c1)*100

# %% [markdown]
# RLPMvsLSTM - polt

# %%
#plt.axis([0,60, 0,17])
x_lables = [0,10,20,30,40,50,60]
plt.plot(real_data_hourly[0], color='c')
plt.plot(predicted_RL[0], color='r')
plt.plot(LSTM_SFC_preds[0], color='g')
plt.legend(['Actual SFC traffic','RLPM','Conventional LSTM Predictor'])
plt.xlabel('minute')
plt.xticks(x_lables)
plt.ylabel('workload')
plt.savefig('RLPMvsLSTM.png', dpi=1000)
plt.show()


# %% [markdown]
# RLPMvsGRU - polt

# %%
#plt.axis([0,60, 0,17])
x_lables = [0,10,20,30,40,50,60]
plt.plot(real_data_hourly[0], color='c')
plt.plot(predicted_RL[0], color='r')
plt.plot(GRU_SFC_preds[0], color='g')
plt.legend(['Actual SFC traffic','RLPM','Conventional GRU Predictor'])
plt.xlabel('minute')
plt.xticks(x_lables)
plt.ylabel('workload')
plt.savefig('RLPMvsGRU.png', dpi=1000)
plt.show()


# %%
x1_lables = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.plot(total_rewards_train)
plt.xlabel('Episodes')
plt.xticks(x1_lables)
plt.ylabel('Total reward')
plt.savefig('total_rewards2.png', dpi=1000, transparent=True)

# %% [markdown]
# Evaluate PMs

# %%
sub1_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC_sub1 = sub1.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    sub1_preds.append(pred_SFC_sub1)

sub1_preds = np.array(sub1_preds).reshape(1,-1)

# %%
sub2_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC_sub2 = sub2.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    sub2_preds.append(pred_SFC_sub2)

sub2_preds = np.array(sub2_preds).reshape(1,-1)

# %%
sub3_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC_sub3 = sub3.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    sub3_preds.append(pred_SFC_sub3)

sub3_preds = np.array(sub3_preds).reshape(1,-1)

# %%
sub4_preds = []
for i in range(len(real_data_hourly[0])):
    pred_SFC_sub4 = sub4.predict(real_data_hourly[0,i:i+4].reshape(1,-1))
    sub4_preds.append(pred_SFC_sub4)

sub4_preds = np.array(sub4_preds).reshape(1,-1)

# %%
def MSE(real, predicted):
    mse_total = 0
    for i in range(int(len(real[0]))):
        mse_total += mean_squared_error([real[0,i]], [predicted[0,i]])
    return mse_total

# %%
mse1 = MSE(real_data_hourly, sub1_preds)
mse2 = MSE(real_data_hourly, sub2_preds)
mse3 = MSE(real_data_hourly, sub3_preds)
mse4 = MSE(real_data_hourly, sub4_preds)
mse_rl = MSE(real_data_hourly, predicted_RL)
mse_lstm = MSE(real_data_hourly, LSTM_SFC_preds)
mse_gru = MSE(real_data_hourly, GRU_SFC_preds)

# %%
print(mse1)
print(mse2)
print(mse3)
print(mse4)
print(mse_rl)
print(mse_lstm)
print(mse_gru)

# %%
mse_values = [mse1, mse2, mse3, mse4, mse_rl]
models = ['FW.GRU', 'DPI.GRU', 'IDS.LSTM', 'LB.GRU', 'RLPM']

# Define different colors for each bar
#colors = ['blue', 'green', 'orange', 'red', '#800080']
colors = ['gray'] * len(models)

# Create a bar graph
plt.bar(models, mse_values, color=colors)

plt.xlabel('Predictor')
plt.ylabel('Hourly Prediction Error')
#plt.title('Performance Evaluation: RLPM vs Individual PMs')
plt.subplots_adjust(bottom=0.3)

plt.savefig('RLPMvsPMs.png', dpi=1080, bbox_inches='tight')
plt.show()


# %%
mse_vals = [mse_lstm, mse_gru, mse_rl]
conv_models = ['LSTM', 'GRU', 'RLPM']

colors = ['gray'] * len(conv_models)

# Create a bar graph
plt.bar(conv_models, mse_vals, color=colors)

plt.xlabel('Predictor')
plt.ylabel('Hourly Prediction Error (MSE)')
#plt.title('Performance Evaluation: RLPM vs Individual PMs')
plt.subplots_adjust(bottom=0.3)

plt.savefig('RLPMvsConventionals.png', dpi=1080, bbox_inches='tight')
plt.show()
