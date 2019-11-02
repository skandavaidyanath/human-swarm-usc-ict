# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:40:38 2019

@author: svaidyanath
"""

from ss_env_v2 import SearchStrategiesEnv
import numpy as np
import time


env = SearchStrategiesEnv()
total_episodes = 10000
max_steps = 50
nS = 87890625   #number of states
nA = 5          #number of actions
w = np.zeros((12,1))    #loc, 9 grid, rel_loc, action
gamma = 0.95
epsilon = 1.
alpha = 0.5



def policy(state):
  A = np.ones(nA, dtype=float) * epsilon / nA
  best_action = np.argmax(np.array([np.matmul(x(state,a).T, w) for a in range(nA)]))
  A[best_action] += (1.0 - epsilon)
  return A


def x(state, action):
  loc, grid, rel_loc = env.decode(state)
  flat_grid = []
  flat_grid.extend(grid[0])
  flat_grid.extend(grid[1])
  flat_grid.extend(grid[2])
  x_sa = [loc]
  x_sa.extend(flat_grid)
  x_sa.extend(rel_loc)
  x_sa.extend(action)
  return np.expand_dims(np.array(x_sa), axis=1)


def error(td_target, previous):
  return (td_target - previous)**2



for episode_num in range(1, total_episodes + 1):
  t = 0
  episode = []
  state = env.reset()
  returns = 0  
  
  while t < max_steps:
    probs = policy(state)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    next_state, reward, done, _ = env.step(action)
    returns += reward
    episode.append((state, action, reward, next_state))
    t += 1
    print(t)
    if done:
      break
    state = next_state
  
  episode_error = 0.
  i = 0
  for state, action, reward, next_state in episode:
    td_target = reward + gamma * np.max(np.array([np.matmul(x(next_state, a).T,w) for a in range(nA)]))
    previous = np.matmul(x(state, action).T, w)
    w += alpha * (td_target - previous) * x(state,action)
    episode_error += error(td_target, previous)
    i += 1
    
  time.sleep(0.1)
  if episode_num % 100 == 0 and episode_num < 8000 and epsilon > 0.01:
    epsilon -= 0.012375
  if episode_num == 8000:
    epsilon = 0.01
  if episode_num == 9000:
      epsilon = 0.
  
  mean_episode_error = episode_error/i
  print('EPISODE {} COMPLETED'.format(episode_num))
  print('MEAN EPISODE ERROR = {}'.format(mean_episode_error))

