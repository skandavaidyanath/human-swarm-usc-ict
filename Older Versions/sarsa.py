# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:31:28 2019

@author: svaidyanath
"""

from rs_env import RSHumanSwarmEnv
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns


groups = ['stubborn_couple', 'babysitter', 'old_couple']

epsilon = 0.95
total_episodes = 10000
max_steps = 100  
lr_rate = 0.80   # alpha
gamma = 0.90

phase = 4


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
      
      
Q = init_q(RSHumanSwarmEnv().observation_space.n, RSHumanSwarmEnv().action_space.n)
    

def choose_action(state):
  '''
  Be epsilon greedy and choose the best action in the given state
  '''
  action=0
  if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()
  else:
    action = np.argmax(Q[state, :])
  return action
  

def learn(state, state2, reward, action, action2):
  '''
  The SARSA update equation
  '''
  predict = Q[state, action]
  target = reward + gamma * Q[state2, action2]
  Q[state, action] = Q[state, action] + lr_rate * (target - predict)
    
  

def training_plot(values):
  sns.set()
  df = pd.DataFrame(values)
  sns_plot = sns.lmplot(x='Episode', y='Returns', col='Type', row='Initial Dead Time', hue='Saved', ci=None, robust=True, truncate=True, scatter_kws={'s': 0.1}, data=df)
  sns_plot.savefig('SARSA Training Curve({}).png'.format(phase))
  
  
def plot_heatmap(values):
  sns.set()
  df = pd.DataFrame(values)
  df1 = df.groupby(['Type', 'Initial Dead Time', 'Saved'], as_index=False).count()
  df2 = df.groupby(['Type', 'Initial Dead Time'], as_index=False).count()
  df['Saved ratio'] 
  df1.drop(df1.index[df1['Saved'] == 'no'], inplace=True)
  df3 = df1
  df3['Saved ratio'] = df1['Episode']/df2['Episode']
  df3 = df3.pivot('Initial Dead Time', 'Type', 'Saved ratio')
  df3 = df3.fillna(0)
  df3 = df3.astype(int)
  sns_plot = sns.heatmap(df3, annot=True, fmt='d', linewidths=0.5)
  fig = sns_plot.get_figure()
  fig.savefig('SARSA Training Statistics({}).png'.format(phase))



statistics = {'Episode':[i for i in range(1, total_episodes + 1)], 'Returns':[], 'Type':[], 'Initial Dead Time': [], 'Saved':[]}

# Start learning by simulating multiple episodes
for episode in range(1, total_episodes + 1):
  t = 0
  index = np.random.randint(0, 3)
  statistics['Type'].append(groups[index])
  env = RSHumanSwarmEnv(group_type=groups[index])
  state = env.reset()
  statistics['Initial Dead Time'].append(env.decode(state)[2])
  action = choose_action(state)
  returns = 0 
  
  while t < max_steps:
    state2, reward, done, info = env.step(action)
    returns += reward  
    
    action2 = choose_action(state2)
    learn(state, state2, reward, action, action2)
    
    state = state2
    action = action2
    t += 1
    print(t)
    
    if done:
      statistics['Returns'].append(returns)
      if reward == 3000:
        statistics['Saved'].append('yes')
      else:
        statistics['Saved'].append('no')
      break
    
    time.sleep(0.1)
  
  if episode % 1000 == 0 and epsilon > 0.05:
    epsilon -= 0.1
  if episode % 2500 == 0 and lr_rate > 0.50:
    lr_rate -= 0.1
    
  print('EPISODE {} COMPLETED'.format(episode))


with open('HSE_SARSA_qTable({}).pkl'.format(phase), 'wb') as fp:
    pickle.dump(Q, fp)
    
training_plot(statistics)
plot_heatmap(statistics)
dataframe = pd.DataFrame(statistics)
dataframe.to_csv('sarsa training info({}).csv'.format(phase), encoding='utf-8')