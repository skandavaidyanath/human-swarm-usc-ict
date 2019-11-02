# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:08:46 2019

@author: svaidyanath
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from new_rs_env import NewRSHumanSwarmEnv
import numpy as np
import time
import pickle

Q = None
num_episodes = 1000
statistics = {'Episode':[i for i in range(1, num_episodes + 1)], 'Returns':[], 'Type':[], 'Initial Dead Time': [], 'Saved':[]}
phase = 28
dead_time = 4  #to initialize the environment


def choose_action(state):
  action = np.argmax(Q[state, :])
  return action



def plot_heatmap(algorithm, values, phase):
  sns.set()
  df = pd.DataFrame(values)
  df1 = df.groupby(['Type', 'Initial Dead Time'], as_index=False).count()
  df1['Saved ratio'] = 0
  for cg in ['stubborn_couple', 'old_couple', 'babysitter']:
    for dt in range(2,5):
      df2 = df.loc[(df['Type'] == cg) & (df['Initial Dead Time'] == dt)]
      df3 = df.loc[(df['Type'] == cg) & (df['Initial Dead Time'] == dt) & (df['Saved'] == 'yes')]
      df1['Saved ratio'].loc[(df1['Type'] == cg) & (df1['Initial Dead Time'] == dt)] = len(df3)/len(df2)
  df1 = df1.pivot('Initial Dead Time', 'Type', 'Saved ratio')
  df1 = df1.fillna(0)
  sns_plot = sns.heatmap(df1, annot=True, fmt='f', linewidths=0.5)
  fig = sns_plot.get_figure()
  fig.savefig('Monte Carlo Test Statistics({}).png'.format(phase))
  
  


def test(algorithm):
  
  file_name = None
  if algorithm == 'Q-Learning':
    file_name = 'HSE_QL_qTable({}).pkl'.format(phase)
  elif algorithm == 'SARSA':
    file_name = 'HSE_SARSA_qTable({}).pkl'.format(phase)
  elif algorithm == 'Expected SARSA':
    file_name = 'HSE_EXPSARSA_qTable({}).pkl'.format(phase)
  elif algorithm == 'Monte Carlo':
    file_name = 'HSE_MC_qTable({}).pkl'.format(phase)
  elif algorithm == 'Off Policy Monte Carlo':
    file_name = 'HSE_OffP_MC_qTable({}).pkl'.format(phase)
  
  groups = ['stubborn_couple', 'babysitter', 'old_couple']


  with open(file_name, 'rb') as fp:
    global Q
    Q = pickle.load(fp)


  fp = open(str(algorithm)+' test logs({}).txt'.format(phase), 'w+')
  # start
  for episode in range(num_episodes):
  
    index = np.random.randint(0, 3)
    env = NewRSHumanSwarmEnv(group_type=groups[index])
    statistics['Type'].append(groups[index])
    state = env.reset()
    statistics['Initial Dead Time'].append(env.decode(state)[2])
    fp.write('Episode: ' + str(episode + 1))
    fp.write('\n')
    fp.write(groups[index])
    fp.write('\n')
    fp.write(env.state_details(state))
    fp.write('\n')
    t = 0
    total_returns = 0
    while t < 100:
      action = choose_action(state) 
      fp.write(env.action_details(action))
      fp.write('\n')
      state2, reward, done, info = env.step(action)  
      total_returns += reward
      state = state2
      fp.write(env.state_details(state))
      fp.write('\n')
      if done:
        if reward == 5000:
          statistics['Saved'].append('yes')
        else:
          statistics['Saved'].append('no')
        break
      time.sleep(0.1)
    statistics['Returns'].append(total_returns)
    
    fp.write('\n')
    fp.write('Episode Return is ' + str(total_returns))
    fp.write('\n')
    fp.write('*************************************')
    fp.write('\n')
    print(episode + 1)
  
  plot_heatmap(algorithm, statistics, phase)  
  fp.close()


    
def main():
  #test('Q-Learning')
  #test('SARSA')
  #test('Expected SARSA')
  test('Monte Carlo')
  #test('Off Policy Monte Carlo')
  df = pd.DataFrame(statistics)
  df.to_csv('monte carlo testing info({}).csv'.format(phase), encoding='utf-8')   #change the name according to the algo you are running
  
  
  
if __name__ == '__main__':
  main()
  
  