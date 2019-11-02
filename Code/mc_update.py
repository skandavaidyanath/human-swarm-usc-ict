# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:36 2019

@author: svaidyanath
"""

from new_rs_env import NewRSHumanSwarmEnv
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns


groups = ['stubborn_couple', 'babysitter', 'old_couple']

epsilon = 0.01
additional_episodes = 500000
max_steps = 100   # max steps in a single episode
gamma = 0.95

phase = 22    # the phase to which you are appending extra episodes
new_phase = 29     # the new phase number
dead_time = 4      # the initial dead time you want for the environment

nS = NewRSHumanSwarmEnv().observation_space.n
nA = NewRSHumanSwarmEnv().action_space.n


def training_plot(values, smoothing_window=10000):
  df = pd.DataFrame(values)
  old_df = pd.read_csv('monte carlo training info({}).csv'.format(phase))
  old_df.drop(columns=['Unnamed: 0'], inplace=True)
  df = pd.concat([old_df, df])
  df = df.reset_index()
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
  plt.savefig('Monte Carlo Training Curve({}).png'.format(new_phase))
  plt.close()
  
  
  
def plot_heatmap(values):
  sns.set()
  df = pd.DataFrame(values)
  old_df = pd.read_csv('monte carlo training info({}).csv'.format(phase))
  old_df.drop(columns=['Unnamed: 0'], inplace=True)
  df = pd.concat([old_df, df])
  df = df.reset_index()
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
  fig.savefig('Monte Carlo Training Statistics({}).png'.format(new_phase))
  


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
    
    
Q = None
with open('HSE_MC_qTable({}).pkl'.format(phase), 'rb') as fp:
    Q = pickle.load(fp)


def policy(state):
  A = np.ones(nA, dtype=float) * epsilon / nA
  best_action = np.argmax(Q[state])
  A[best_action] += (1.0 - epsilon)
  return A
  
  
  
returns_sum = defaultdict(float)
returns_count = defaultdict(float)
statistics = {'Episode':[i for i in range(500001, 1000001)], 'Returns':[], 'Type':[], 'Initial Dead Time': [], 'Saved':[]}

# Start learning by simulating multiple episodes
for episode_num in range(1, additional_episodes + 1):
  t = 0
  episode = []
  index = np.random.randint(0, 3)
  statistics['Type'].append(groups[index])
  env = NewRSHumanSwarmEnv(group_type=groups[index], init_dead_time=dead_time)
  state = env.reset()
  statistics['Initial Dead Time'].append(env.decode(state)[2])
  returns = 0  
  
  while t < max_steps:
    probs = policy(state)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    next_state, reward, done, _ = env.step(action)
    returns += reward
    episode.append((state, action, reward))
    t += 1
    print(t)
    if done:
      statistics['Returns'].append(returns)
      if reward == 5000:
        statistics['Saved'].append('yes')
      else:
        statistics['Saved'].append('no')
      break
    state = next_state
  
  sa_in_episode = set([(x[0], x[1]) for x in episode])
  for state, action in sa_in_episode:
    sa_pair = (state, action)
    # First occurence MC
    first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
    G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
    returns_sum[sa_pair] += G
    returns_count[sa_pair] += 1.0
    Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    
    
  time.sleep(0.1)
  if episode_num == 450000:
    epsilon = 0.
  print('EPISODE {} COMPLETED'.format(episode_num))


with open('HSE_MC_qTable({}).pkl'.format(new_phase), 'wb') as fp:
    pickle.dump(Q, fp)
    
    
dataframe = pd.DataFrame(statistics)
dataframe.to_csv('monte carlo training info({}).csv'.format(new_phase), encoding='utf-8')
training_plot(statistics)
plot_heatmap(statistics)
