# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:39:31 2019

@author: svaidyanath
"""

from rs_env import RS_HumanSwarmEnv
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns



groups = ['stubborn_couple', 'babysitter', 'old_couple']

epsilon = 0.10
total_episodes = 10000
max_steps = 100
gamma = 0.90

phase = 4


nS = RS_HumanSwarmEnv().observation_space.n
nA = RS_HumanSwarmEnv().action_space.n



def training_plot(values):
  sns.set()
  df = pd.DataFrame(values)
  sns_plot = sns.lmplot(x='Episode', y='Returns', col='Type', row='Initial Dead Time', hue='Saved', ci=None, robust=True, truncate=True, scatter_kws={'s': 0.1}, data=df)
  sns_plot.savefig('Off Policy MC Training Curve({}).png'.format(phase))
  
  
  
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
  fig.savefig('Off Policy MC Training Statistics({}).png'.format(phase))




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
    

Q = init_q(nS, nA)
C = init_q(nS, nA)


def target_policy(state):
  A = np.ones(nA, dtype=float) * epsilon / nA
  best_action = np.argmax(Q[state])
  A[best_action] += (1.0 - epsilon)
  return A


def behaviour_policy(state):
  A = np.ones(nA, dtype=float) / nA
  return A
  


statistics = {'Episode':[i for i in range(1, total_episodes + 1)], 'Returns':[], 'Type':[], 'Initial Dead Time': [], 'Saved':[]}


# Start learning by simulating multiple episodes
for episode_num in range(total_episodes):
  t = 0
  episode = []
  index = np.random.randint(0, 3)
  statistics['Type'].append(groups[index])
  env = RS_HumanSwarmEnv(group_type=groups[index])
  state = env.reset()
  statistics['Initial Dead Time'].append(env.decode(state)[2])
  returns = 0  
  
  while t < max_steps:
    probs = behaviour_policy(state)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    next_state, reward, done, _ = env.step(action)
    returns += reward
    episode.append((state, action, reward))
    if done:
      statistics['Returns'].append(returns)
      if reward == 3000:
        statistics['Saved'].append('yes')
      else:
        statistics['Saved'].append('no')
      break
    state = next_state
    t += 1
    print(t)
    
  G = 0.0
  W = 1.0
  
  for t in range(len(episode))[::-1]:
    state, action, reward = episode[t]
    G = gamma * G + reward
    C[state][action] += W
    Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
    if action !=  np.argmax(target_policy(state)):
      break
    W = W * 1./behaviour_policy(state)[action]
     
  time.sleep(0.1)
  print('EPISODE {} COMPLETED'.format(episode_num + 1))


with open('HSE_OffP_MC_qTable({}).pkl'.format(phase), 'wb') as fp:
    pickle.dump(Q, fp)
  
  
training_plot(statistics)
plot_heatmap(statistics)
df = pd.DataFrame(statistics)
df.to_csv('off policy mc training info({}).csv'.format(phase), encoding='utf-8')
    