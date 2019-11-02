# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:53:49 2019

@author: svaidyanath
"""

from new_rs_env import NewRSHumanSwarmEnv
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns


groups = ['stubborn_couple', 'babysitter', 'old_couple']

epsilon = 0.95
total_episodes = 200000
max_steps = 100    #in an episode
lr_rate = 1.    #alpha
gamma = 0.95

phase = 18
dead_time = 4    #to initialize the environment


def init_q(s, a, type='zeros'):
    '''
    Initialise the Q table with ones, random values or zeros.
    '''
    if type == 'ones':
        return np.ones((s, a))
    elif type == 'random':
        return np.random.random((s, a))
    elif type == 'zeros':
        return np.zeros((s, a))
    

Q = init_q(NewRSHumanSwarmEnv().observation_space.n, NewRSHumanSwarmEnv().action_space.n)




def choose_action(state):
  '''
  Be epsilon greedy and choose the best action in the given state
  '''
  action = 0
  if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()
  else:
    action = np.argmax(Q[state, :])
  return action
  


def learn(state, state2, reward, action):
  '''
  The q-learning update equation
  '''
  predict = Q[state, action]
  target = reward + gamma * np.max(Q[state2, :])
  Q[state, action] = Q[state, action] + lr_rate * (target - predict)
  
  
  

  
def training_plot(values, smoothing_window=2000):
  df = pd.DataFrame(values)
  plt.figure(figsize=(30,10))
  i = 131
  for cg in ['stubborn_couple', 'old_couple', 'babysitter']:
    df1 = df.loc[(df['Type'] == cg) & (df['Initial Dead Time'] == dead_time)]
    rewards_smoothed = pd.Series(df1['Returns']).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.subplot(i)
    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.title('Initial Dead Time = {}, Type = {}'.format(dead_time, cg))
    plt.plot(rewards_smoothed)
    i += 1
  plt.savefig('Q-Learning Training Curve({}).png'.format(phase))
  
  
  
def plot_heatmap(values):
  sns.set()
  df = pd.DataFrame(values)
  df1 = df.groupby(['Type', 'Initial Dead Time'], as_index=False).count()
  df1['Saved ratio'] = 0
  for cg in ['stubborn_couple', 'old_couple', 'babysitter']:
    df2 = df.loc[(df['Type'] == cg) & (df['Initial Dead Time'] == dead_time)]
    df3 = df.loc[(df['Type'] == cg) & (df['Initial Dead Time'] == dead_time) & (df['Saved'] == 'yes')]
    df1['Saved ratio'].loc[(df1['Type'] == cg) & (df1['Initial Dead Time'] == dead_time)] = len(df3)/len(df2)
  df1 = df1.pivot('Initial Dead Time', 'Type', 'Saved ratio')
  df1 = df1.fillna(0)
  sns_plot = sns.heatmap(df1, annot=True, fmt='f', linewidths=0.5)
  fig = sns_plot.get_figure()
  fig.savefig('Q-Learning Training Statistics({}).png'.format(phase))
  
  
statistics = {'Episode':[i for i in range(1, total_episodes + 1)], 'Returns':[], 'Type':[], 'Initial Dead Time': [], 'Saved':[]}


# Start learning by simulating multiple episodes
for episode in range(1, total_episodes + 1):
  t = 0
  index = np.random.randint(0, 3)
  statistics['Type'].append(groups[index])
  env = NewRSHumanSwarmEnv(group_type=groups[index], init_dead_time=dead_time)
  state = env.reset()
  statistics['Initial Dead Time'].append(env.decode(state)[2])
  returns = 0  
  
  while t < max_steps:
    action = choose_action(state)  
    state2, reward, done, info = env.step(action)
    returns += reward
      
    learn(state, state2, reward, action)
    
    state = state2
    t += 1
    print(t)
    
    if done:
      statistics['Returns'].append(returns)
      if reward == 5000:
        statistics['Saved'].append('yes')
      else:
        statistics['Saved'].append('no')
      break
    
    time.sleep(0.1)
    
  if episode % 2000 == 0 and epsilon > 0.01:
    epsilon -= 0.0094
    if epsilon < 0:
      epsilon = 0.01
  lr_rate = 1/(1 + episode)
    
  print('EPISODE {} COMPLETED'.format(episode))


with open('HSE_QL_qTable({}).pkl'.format(phase), 'wb') as fp:
    pickle.dump(Q, fp)
    
    
training_plot(statistics)
plot_heatmap(statistics)
dataframe = pd.DataFrame(statistics)
dataframe.to_csv('q-learning training info({}).csv'.format(phase), encoding='utf-8')


    
  