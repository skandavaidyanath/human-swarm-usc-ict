# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:39:59 2019

@author: svaidyanath
"""

from gym.envs.toy_text import discrete
import numpy as np
from itertools import combinations
import copy


class SearchStrategiesEnv(discrete.DiscreteEnv):
  '''
  Class that will simulate the environment of the Seacrh Trategies required for the Human Swarm Interaction Project
  For detailed description of the problem, read the accompanying docs. This is for a specific case of a 3 * 3 block with two drones
  The state variables are:
    1) Current location of drone
    2) Status information of each of the squares in the block
    3) Relative location of second drone
  There are 87890625 states
  The action space is:
    1) 0 - Stay
    2) 1 - Move North
    3) 2 - Move East
    4) 3 - Move West
    5) 4 - Move South
  '''
  
  def __init__(self, drone_entry_point=1, fire_direction='south'):
    
    num_states = 87890625
    num_actions = 5
    
    self.single_burn_time_update_prob = 0.25
    self.double_burn_time_update_prob = 0.15
    self.triple_burn_time_update_prob = 0.1
    self.no_burn_time_update = 1 - (self.single_burn_time_update_prob + self.double_burn_time_update_prob + self.triple_burn_time_update_prob)
    
    '''
    The initial state distribution.
    We allow only states where only one quadrant is on fire and all other variables are 0 to be initial states
    '''
    initial_state_distrib = np.zeros(num_states)   
  
    '''
    The transition function. 
    P[s][a] is a list of tuple of four values:
      The probability of the transition
      The state to which we transition
      Reward for the transition
      Done = True or False
    '''
    P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}
    
    '''
    The status of each square. 
    0 - burned
    1,2,3 - burn times
    4 - searched
    '''
    status_options = [0, 1, 2, 3, 4]  
    
    '''
    Relative location of other drone
    0 - not in range/ dropped out
    1 - North
    2 - East
    3 - West
    4 - South
    '''
    other_drone_options = [0, 1, 2, 3, 4]
    
    for loc in range(1,10):
      for status1 in status_options:
        for status2 in status_options:
          for status3 in status_options:
            for status4 in status_options:
              for status5 in status_options:
                for status6 in status_options:
                  for status7 in status_options:
                    for status8 in status_options:
                      for status9 in status_options:
                        for rel_loc in other_drone_options:
                          grid = [[status1, status2, status3], [status4, status5, status6], [status7, status8, status9]]
                          state = self.encode([loc, grid, rel_loc])
                          if loc == drone_entry_point and self.__match(grid, fire_direction, drone_entry_point) and self.__valid(state):
                            initial_state_distrib[state] += 1
                          if self.__valid(state) and not self.__terminal(state):
                            for action in range(num_actions):
                              probs, states, rewards, dones = self.__simulate(state, action)
                            for i in range(len(probs)):
                              P[state][action].append((probs[i], states[i], rewards[i], dones[i]))
                          else:
                            for action in range(num_actions):
                              P[state][action].append((1.0, state, 0, True))
    
    initial_state_distrib /= initial_state_distrib.sum()
    discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)
    
    
  
  def __valid(self, state):
    '''
    Checks if the state is valid.
    A state is invalid iff:
    1) the loc position is not in status 0 or 4
    2) the relative location is an impossible position
    '''
    
    loc, grid, rel_loc = self.decode(state)
    
    if grid[(loc-1)//3][(loc-1)%3] not in [0,4]:
      return False
    
    if loc == 1 and rel_loc in [1,3]:
      # top left
      return False
    if loc == 2 and rel_loc in [1]:
      # top middle
      return False
    if loc == 3 and rel_loc in [1,2]:
      # top right
      return False
    if loc == 4 and rel_loc in [3]:
      # middle left
      return False
    if loc == 6 and rel_loc in [2]:
      # middle right
      return False
    if loc == 7 and rel_loc in [3,4]:
      # bottom left
      return False
    if loc == 8 and rel_loc in [4]:
      # bottom middle
      return False
    if loc == 9 and rel_loc in [2,4]:
      # bottom right
      return False

    return True
  
  
  def __terminal(self, state):
    '''
    A state is terminal iff every position on the grid is 0 or -1 i.e. searched or burned
    '''
    
    loc, grid, rel_loc = self.decode(state)
    
    for i in range(3):
      for j in range(3):
        if grid[i][j] not in [0,4]:
          return False
    
    return True
  
  
  def __match(self, grid, fire_direction, drone_entry_point):
    '''
    To check if the grid matches the initial conditions specified by the user
    '''
    
    initial_grid = [[0,0,0] for i in range(3)]
    
    if fire_direction == 'north':
      initial_grid[0] = [1,1,1]
      initial_grid[1] = [2,2,2]
      initial_grid[2] = [3,3,3]
    if fire_direction == 'east':
      initial_grid[0][2], initial_grid[1][2], initial_grid[2][2] = [1,1,1]
      initial_grid[0][1], initial_grid[1][1], initial_grid[2][1] = [2,2,2]
      initial_grid[0][0], initial_grid[1][0], initial_grid[2][0] = [3,3,3]
    if fire_direction == 'west':
      initial_grid[0][0], initial_grid[1][0], initial_grid[2][0] = [1,1,1]
      initial_grid[0][1], initial_grid[1][1], initial_grid[2][1] = [2,2,2]
      initial_grid[0][2], initial_grid[1][2], initial_grid[2][2] = [3,3,3]
    if fire_direction == 'south':
      initial_grid[2] = [1,1,1]
      initial_grid[1] = [2,2,2]
      initial_grid[0] = [3,3,3]
    
    initial_grid[0][drone_entry_point-1] = 4
    
    return grid == initial_grid
      
  
  
  def encode(self, state):
    '''
    Function encodes the state into a number
    '''
    
    loc, grid, rel_loc = state
    row1, row2, row3 = grid
    status1, status2, status3 = row1
    status4, status5, status6 = row2
    status7, status8, status9 = row3
    
    i = loc
    i *= 5
    i += status1
    i *= 5
    i += status2
    i *= 5
    i += status3
    i *= 5
    i += status4
    i *= 5
    i += status5
    i *= 5
    i += status6
    i *= 5
    i += status7
    i *= 5
    i += status8
    i *= 5
    i += status9
    i *= 5
    i += rel_loc
    
    return i
  
  
  
  def decode(self, i):
    '''
    Function decodes the number into a state
    '''
    
    out = []
    out.append(i % 5)  #rel_loc
    i = i // 5
    out.append(i % 5)  #status9
    i = i // 5
    out.append(i % 5) #status8
    i = i // 5
    out.append(i % 5) #status7
    i = i // 5
    out.append(i % 5) #status6
    i = i // 5
    out.append(i % 5) #status5
    i = i // 5
    out.append(i % 5) #status4
    i = i // 5
    out.append(i % 5) #status3
    i = i // 5
    out.append(i % 5) #status2
    i = i // 5
    out.append(i % 5) #status1
    i = i // 5
    out.append(i) #loc
    i = i // 9
    assert i == 0
    
    out.reverse()
    loc = out[0]
    row1 = out[1:4]
    row2 = out[4:7]
    row3 = out[7:10]
    grid = [row1, row2, row3]
    rel_loc = out[10]
    
    return [loc, grid, rel_loc]
  
  
  
  def state_details(self, state):
    '''
    Gives explicit details about the state 
    Utility function useful for debugging
    '''
    loc, grid, rel_loc = self.decode(state)
    
    state_info = 'The location of the drone is {}'.format(loc)
    state_info += '\n'
    
    for i in range(3):
      for j in range(3):
        state_info += str(grid[i][j])
        state_info += ' '
      state_info += '\n'
    
    state_info = 'The relative location of the other drone is {}'.format(rel_loc)
    state_info += '\n'
    
    return state_info
    
  
  
  def action_details(self, action):
    '''
    Gives details about the action
    Utility function useful for debugging
    '''
    
    if action == 0:
      return 'STAY\n'
    if action == 1:
      return 'NORTH\n'
    if action == 2:
      return 'EAST\n'
    if action == 3:
      return 'WEST\n'
    if action == 4:
      return 'SOUTH\n'
    
    
  def __possible_rel_locs(self, loc):
    '''
    Gives the possible rel_locs for each location of the drone
    Utility function
    '''
    if loc == 1:
      return [0, 2, 4]
    if loc == 2:
      return [0, 2, 3, 4]
    if loc == 3:
      return [0, 3, 4]
    if loc == 4:
      return [0, 1, 2, 4]
    if loc == 5:
      return [0, 1, 2, 3, 4]
    if loc == 6:
      return [0, 1, 3, 4]
    if loc == 7:
      return [0, 1, 2]
    if loc == 8:
      return [0, 1, 2, 3]
    if loc == 9:
      return [0, 1, 3]
    
    
    
  def __simulate(self, state, action):
    '''
    Simulates the action at the given state and returns the possible transitions
    '''
    
    loc, grid, rel_loc = self.decode(state)
    probs = []
    states = []
    rewards = []
    dones = []
    
    if action == 0:
      # Stay action
      # Just take care of environment updates
      grid_status_updates = self.__grid_status_updates(loc, grid, rel_loc)
      for transition in grid_status_updates:
        new_state = self.decode(transition['state'])
        possible_rel_locs = self.__possible_rel_locs(loc)
        for new_rel_loc in possible_rel_locs:
          probs.append(transition['prob'] * 1/len(possible_rel_locs))
          new_state[2] = new_rel_loc
          states.append(new_state)
          rewards.append(transition['reward']-1)
          dones.append(self.__terminal(transition['state']))
        
    if action == 1:
      # Move North
      new_loc = loc
      if loc not in [1,2,3]:
        new_loc -= 3
      grid_status_updates = self.__grid_status_updates(loc, grid, rel_loc)
      for transition in grid_status_updates:
        new_state = self.decode(transition['state'])
        new_state[0] = new_loc
        if new_state[1][(new_loc-1)//3][(new_loc-1)%3] != 0:
          # if the location we move to isn't burned at the same time, then we search it
          new_state[1][(new_loc-1)//3][(new_loc-1)%3] = 4
        possible_rel_locs = self.__possible_rel_locs(new_loc)
        for new_rel_loc in possible_rel_locs:
          probs.append(transition['prob'] * 1/len(possible_rel_locs))
          new_state[2] = new_rel_loc
          states.append(new_state)
          rewards.append(transition['reward']-1)
          dones.append(self.__terminal(transition['state']))
        
    if action == 2:
      # Move East
      new_loc = loc
      if loc not in [3,6,9]:
        new_loc += 1
      grid_status_updates = self.__grid_status_updates(loc, grid, rel_loc)
      for transition in grid_status_updates:
        new_state = self.decode(transition['state'])
        new_state[0] = new_loc
        if new_state[1][(new_loc-1)//3][(new_loc-1)%3] != 0:
          # if the location we move to isn't burned at the same time, then we search it
          new_state[1][(new_loc-1)//3][(new_loc-1)%3] = 4
        possible_rel_locs = self.__possible_rel_locs(new_loc)
        for new_rel_loc in possible_rel_locs:
          probs.append(transition['prob'] * 1/len(possible_rel_locs))
          new_state[2] = new_rel_loc
          states.append(new_state)
          rewards.append(transition['reward']-1)
          dones.append(self.__terminal(transition['state']))
        
    if action == 3:
      # Move West
      new_loc = loc
      if loc not in [1,4,7]:
        new_loc -= 1
      grid_status_updates = self.__grid_status_updates(loc, grid, rel_loc)
      for transition in grid_status_updates:
        new_state = self.decode(transition['state'])
        new_state[0] = new_loc
        if new_state[1][(new_loc-1)//3][(new_loc-1)%3] != 0:
          # if the location we move to isn't burned at the same time, then we search it
          new_state[1][(new_loc-1)//3][(new_loc-1)%3] = 4
        possible_rel_locs = self.__possible_rel_locs(new_loc)
        for new_rel_loc in possible_rel_locs:
          probs.append(transition['prob'] * 1/len(possible_rel_locs))
          new_state[2] = new_rel_loc
          states.append(new_state)
          rewards.append(transition['reward']-1)
          dones.append(self.__terminal(transition['state']))
        
    if action == 4:
      # Move South
      new_loc = loc
      if loc not in [7,8,9]:
        new_loc += 3
      grid_status_updates = self.__grid_status_updates(loc, grid, rel_loc)
      for transition in grid_status_updates:
        new_state = self.decode(transition['state'])
        new_state[0] = new_loc
        if new_state[1][(new_loc-1)//3][(new_loc-1)%3] != 0:
          # if the location we move to isn't burned at the same time, then we search it
          new_state[1][(new_loc-1)//3][(new_loc-1)%3] = 4
        possible_rel_locs = self.__possible_rel_locs(new_loc)
        for new_rel_loc in possible_rel_locs:
          probs.append(transition['prob'] * 1/len(possible_rel_locs))
          new_state[2] = new_rel_loc
          states.append(new_state)
          rewards.append(transition['reward']-1)
          dones.append(self.__terminal(transition['state']))
            
    assert round(sum(probs), 14) == 1
    assert (np.array(probs) > 0).all()
    assert len(probs) == len(states) == len(rewards) == len(dones)
    return probs, states, rewards, dones    
  
  
  
  
  def __search_updates(self, loc, grid, rel_loc, out_of_sight_possibilities):
    '''
    Update the search performance of the second drone
    '''
    search_updates = []
    
    if loc == 1:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 2:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 3:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 4:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 5:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 6:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 4:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+3-1)//3][(loc+3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 7:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 8:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 2:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc+1-1)//3][(loc+1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
    if loc == 9:
      if rel_loc == 0:
        for pos in out_of_sight_possibilities:
          transition = {}
          new_grid = copy.deepcopy(grid)
          transition['prob'] = 1/len(out_of_sight_possibilities)
          new_grid[(pos-1)//3][(pos-1)%3] = 4
          new_state = self.encode([loc, new_grid, rel_loc])
          transition['state'] = new_state
          transition['reward'] = 0
          search_updates.append(transition)
      if rel_loc == 1:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-3-1)//3][(loc-3-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
      if rel_loc == 3:
        transition = {}
        new_grid = copy.deepcopy(grid)
        transition['prob'] = 1.
        new_grid[(loc-1-1)//3][(loc-1-1)%3] = 4
        new_state = self.encode([loc, new_grid, rel_loc])
        transition['state'] = new_state
        transition['reward'] = 0
        search_updates.append(transition)
        
        
    probs_sum = 0
    for transition in search_updates:
      assert transition['prob'] > 0
      probs_sum += transition['prob']
    assert round(probs_sum, 15) == 1
    return search_updates
  
  
  def __single_square_updates(self, update):
    '''
    Updates the burn time of a single square
    '''
    loc, grid, rel_loc = self.decode(update['state'])
    single_square_updates = []
    possibilities = []
    for i in range(3):
      for j in range(3):
        if grid[i][j] not in [0,4]:
          possibilities.append(3*i + j + 1)
    
    for pos in possibilities:
      transition = {}
      new_grid = copy.deepcopy(grid)
      transition['prob'] = 1/len(possibilities) * self.single_burn_time_update_prob * update['prob']
      new_grid[(pos-1)//3][(pos-1)%3] -= 1
      if new_grid[(pos-1)//3][(pos-1)%3] == 0:
        transition['reward'] = -100
      else:
        transition['reward'] = 0
      new_state = self.encode([loc, new_grid, rel_loc])
      transition['state'] = new_state
      single_square_updates.append(transition)
      
    return single_square_updates
  
  
  
  def __double_square_updates(self, update):
    loc, grid, rel_loc = self.decode(update['state'])
    double_square_updates = []
    possibilities = []
    for i in range(3):
      for j in range(3):
        if grid[i][j] not in [0,4]:
          possibilities.append(3*i + j + 1)
    
    comb = list(combinations(possibilities, 2))
    for tup in comb:
      pos1, pos2 = tup
      transition = {}
      new_grid = copy.deepcopy(grid)
      transition['prob'] = 1/len(comb) * self.double_burn_time_update_prob * update['prob']
      new_grid[(pos1-1)//3][(pos1-1)%3] -= 1
      new_grid[(pos2-1)//3][(pos2-1)%3] -= 1
      if new_grid[(pos1-1)//3][(pos1-1)%3] == 0 and new_grid[(pos2-1)//3][(pos2-1)%3] == 0:
        transition['reward'] = -200
      elif new_grid[(pos1-1)//3][(pos1-1)%3] == 0:
        transition['reward'] = -100
      elif new_grid[(pos2-1)//3][(pos2-1)%3] == 0:
        transition['reward'] = -100
      else:
        transition['reward'] = 0
      new_state = self.encode([loc, new_grid, rel_loc])
      transition['state'] = new_state
      double_square_updates.append(transition)
      
    return double_square_updates
  
  
  def __triple_square_updates(self, update):
    loc, grid, rel_loc = self.decode(update['state'])
    triple_square_updates = []
    possibilities = []
    for i in range(3):
      for j in range(3):
        if grid[i][j] not in [0,4]:
          possibilities.append(3*i + j + 1)
    
    comb = list(combinations(possibilities, 3))
    for tup in comb:
      pos1, pos2, pos3 = tup
      transition = {}
      new_grid = copy.deepcopy(grid)
      transition['prob'] = 1/len(comb) * self.triple_burn_time_update_prob * update['prob']
      new_grid[(pos1-1)//3][(pos1-1)%3] -= 1
      new_grid[(pos2-1)//3][(pos2-1)%3] -= 1
      new_grid[(pos3-1)//3][(pos3-1)%3] -= 1
      if new_grid[(pos1-1)//3][(pos1-1)%3] == 0 and new_grid[(pos2-1)//3][(pos2-1)%3] == 0 and new_grid[(pos3-1)//3][(pos3-1)%3] == 0:
        transition['reward'] = -300
      elif new_grid[(pos1-1)//3][(pos1-1)%3] == 0 and new_grid[(pos2-1)//3][(pos2-1)%3] == 0:
        transition['reward'] = -200
      elif new_grid[(pos1-1)//3][(pos1-1)%3] == 0 and new_grid[(pos3-1)//3][(pos3-1)%3] == 0:
        transition['reward'] = -200
      elif new_grid[(pos2-1)//3][(pos2-1)%3] == 0 and new_grid[(pos3-1)//3][(pos3-1)%3] == 0:
        transition['reward'] = -200
      elif new_grid[(pos1-1)//3][(pos1-1)%3] == 0:
        transition['reward'] = -100
      elif new_grid[(pos2-1)//3][(pos2-1)%3] == 0:
        transition['reward'] = -100
      elif new_grid[(pos3-1)//3][(pos3-1)%3] == 0:
        transition['reward'] = -100
      else:
        transition['reward'] = 0
      new_state = self.encode([loc, new_grid, rel_loc])
      transition['state'] = new_state
      triple_square_updates.append(transition)
      
    return triple_square_updates
  
   
  
  def __grid_status_updates(self, loc, grid, rel_loc):
    '''
    Environment related updates i.e. updating the status of the squares
    '''
    grid_status_updates = []
    
    out_of_sight_possibilities = None
    if loc == 1:
      out_of_sight_possibilities = [3,5,6,7,8,9]
    if loc == 2:
      out_of_sight_possibilities = [4,6,7,8,9]
    if loc == 3:
      out_of_sight_possibilities = [1,4,5,7,8,9]
    if loc == 4:
      out_of_sight_possibilities = [1,2,3,7,8,9]
    if loc == 5:
      out_of_sight_possibilities = [1,3,7,9]
    if loc == 6:
      out_of_sight_possibilities = [1,2,4,7,8]
    if loc == 7:
      out_of_sight_possibilities = [1,2,3,5,6,9]
    if loc == 8:
      out_of_sight_possibilities = [1,2,3,4,6]
    if loc == 9:
      out_of_sight_possibilities = [1,2,3,4,5,7]
    
    grid_copy = copy.deepcopy(grid)
    search_updates = self.__search_updates(loc, grid_copy, rel_loc, out_of_sight_possibilities)
    
    for update in search_updates:
      grid_status_updates.extend(self.__single_square_updates(update))
      grid_status_updates.extend(self.__double_square_updates(update))
      grid_status_updates.extend(self.__triple_square_updates(update))
      grid_status_updates.append({'prob':self.no_burn_time_update * update['prob'], 'state':update['state'], 'reward':0})
      
    probs_sum = 0
    for transition in grid_status_updates:
      assert transition['prob'] > 0
      probs_sum += transition['prob']
    assert round(probs_sum, 15) == 1
    
    return grid_status_updates
    
    
    
    
    
        
    
      
    
    
      