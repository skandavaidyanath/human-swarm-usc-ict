# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:36:33 2019

@author: svaidyanath
"""

from gym.envs.toy_text import discrete
import numpy as np
import copy





class HumanSwarmEnv(discrete.DiscreteEnv):
  '''
  Class that will describe the environment of the Human Swarm Interaction. For detailed description of the problem, read the accompanying docs.
  The state variables are:
    For 3 drones, there are 1185408 states.
    [x,x,x,x] - 4 bit vector describing the fire locations. Position 0 is top left quadrant, 1 is top right, 2 is bottom left, 3 is bottom right
    [x,x,x] - status values for 3 drones. Possible status conditions range from 0-7. Read the docs
    [[x,y], [x,y], [x,y]] - engagement variables for 3 drones. 0<=x<=2, 0<=y<=1. x is number of warnings in current engagement, y is number of SP negotiations
  The action space is:
    There are 14 possible actions:
      0 - NOP
      1 - Interrupt Operator
      2 - Drone 1 Guide
      3 - Drone 2 Guide
      4 - Drone 3 Guide
      5 - Drone 1 Engage
      6 - Drone 2 Engage
      7 - Drone 3 Engage
      8 - Drone 1 Warn
      9 - Drone 2 Warn
      10 - Drone 3 Warn
      11 - Drone 1 SP Negotiate
      12 - Drone 2 SP Negotiate
      13 - Drone 3 SP Negotiate
  '''
  
  def __init__(self, fire_update_probability = 0.0625, operator_play_probability=0.33, num_groups_to_save=3):
    self.fire_update_probability = fire_update_probability
    self.operator_play_probability = operator_play_probability   # we update the state with a random operator play that changes the state 'randomly' after a certain no. of 3 state ticks
    self.num_groups_to_save = num_groups_to_save
    self.initial_people_locations = np.zeros(self.num_groups_to_save)  # has size of num_groups_to_save and each value can be 0,1,2,3 representing the quadrant numbers
    self.ease_index = np.zeros(self.num_groups_to_save)

    
    num_states = 1769472
    num_actions = 14
    
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
      Done = 1 or 0 
    '''
    P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}
    
    for tl in range(2):
      for tr in range(2):
        for bl in range(2):
          for br in range(2):
            for s1 in range(8):
              for s2 in range(8):
                for s3 in range(8):
                  for e1w in range(3):
                    for e1n in range(2):
                      for e2w in range(3):
                        for e2n in range(2):
                          for e3w in range(3):
                            for e3n in range(2):
                              fire_data = [tl, tr, bl, br]
                              status_data = [s1, s2, s3]
                              engagement_data = [[e1w, e1n], [e2w, e2n], [e3w, e3n]]
                              state = self.encode([fire_data, status_data, engagement_data])
                              if s1 == s2 == s3 == e1w == e1n == e2w == e2n == e3w == e3n == 0:
                                if tl ==1 and tr == bl == br == 0:
                                  initial_state_distrib[state] += 1
                                elif tr ==1 and tl == bl == br == 0:
                                  initial_state_distrib[state] += 1
                                elif bl ==1 and tl == tr == br == 0:
                                  initial_state_distrib[state] += 1
                                elif br ==1 and tl == tr == bl == 0:
                                  initial_state_distrib[state] += 1
                              for action in range(num_actions):
                                probs, new_states, rewards, dones = self.simulate(state, action)
                                for i in range(len(probs)):
                                  P[state][action].append((probs[i], new_states[i], rewards[i], dones[i]))
                                
    initial_state_distrib /= initial_state_distrib.sum()
    discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)
    
    
    
  def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        state = self.decode(self.s)
        
        fire_data = state[0]
        quadrants = [i for i in range(4)]        
        quadrants.remove(fire_data.index(1))
        for i in range(self.num_groups_to_save):
            self.initial_people_locations[i] = np.random.choice(quadrants)
            self.ease_index[i] = np.random.uniform(0,1)
        
        self.lastaction = None
        return self.s
    
    
    
  def encode(self, state):
    '''
    Function encodes the state into a number
    tl = top right quadrant
    tr = top left quadrant
    bl = bottom right quadrant
    br = bottom left quadrant
    s1 = drone 1 status
    s2 = drone 2 status
    s3 = drone 3 status
    e1w = engagement warning variable drone 1
    e1n = engagement negotiation variable drone 1
    e2w = engagement warning variable drone 2
    e2n = engagement negotiation variable drone 2
    e3w = engagement warning variable drone 3
    e3n = engagement negotiation variable drone 3
    '''
    fire_data, status_data, engagement_data = state
    tl, tr, bl, br = fire_data
    s1, s2, s3 = status_data
    e1, e2, e3 = engagement_data
    e1w, e1n = e1
    e2w, e2n = e2
    e3w, e3n = e3
    i = tl
    i *= 2
    i += tr
    i *= 2
    i += bl
    i *= 2
    i += br
    i *= 8
    i += s1
    i *= 8
    i += s2
    i *= 8
    i += s3
    i *= 3
    i += e1w
    i *= 2
    i += e1n
    i *= 3
    i += e2w
    i *= 2
    i += e2n
    i *= 3
    i += e3w
    i *= 2
    i += e3n
    return i



  def decode(self, i):
    '''
    Function decodes the number into a state
    tl = top right quadrant
    tr = top left quadrant
    bl = bottom right quadrant
    br = bottom left quadrant
    s1 = drone 1 status
    s2 = drone 2 status
    s3 = drone 3 status
    e1w = engagement warning variable drone 1
    e1n = engagement negotiation variable drone 1
    e2w = engagement warning variable drone 2
    e2n = engagement negotiation variable drone 2
    e3w = engagement warning variable drone 3
    e3n = engagement negotiation variable drone 3
    '''
    out = []
    out.append(i % 2)  #e3n
    i = i // 2
    out.append(i % 3)  #e3w
    i = i // 3
    out.append(i % 2)  #e2n
    i = i // 2
    out.append(i % 3)  #e2w
    i = i // 3
    out.append(i % 2)  #e1n
    i = i // 2
    out.append(i % 3)  #e1w
    i = i // 3
    out.append(i % 8)  #s3
    i = i // 8
    out.append(i % 8)  #s2
    i = i // 8
    out.append(i % 8)  #s1
    i = i // 8
    out.append(i % 2)  #br
    i = i // 2
    out.append(i % 2)  #bl
    i = i // 2
    out.append(i % 2)  #tr
    i = i // 2
    out.append(i)  #tl
    i = i // 2
    assert i == 0
    out.reverse()
    fire_data = [out[0], out[1], out[2], out[3]]
    status_data = [out[4], out[5], out[6]]
    engagement_data = [[out[7], out[8]], [out[9], out[10]], [out[11], out[12]]]
    return [fire_data, status_data, engagement_data]
  

  
  def simulate(self, state, action):
    fire_data, status_data, engagement_data = self.decode(state)
    
    fire_updates = self.update_fire_data(fire_data)
    
    if action == 0:
      # NOP ACTION
      # We have plays from the environment and/or from the operator
      probs = []
      new_states = []
      rewards = []
      dones = []
      environment_reachable_transitions = self.simulate_environment(fire_data, status_data, engagement_data)
      operator_reachable_transitions = self.simulate_operator(fire_data, status_data, engagement_data)
      for i in range(len(environment_reachable_transitions)):
        probs.append(0.5 * (1 - self.fire_update_probability) * environment_reachable_transitions[i]['prob'])
        new_states.append(environment_reachable_transitions[i]['state'])
        rewards.append(environment_reachable_transitions[i]['reward'])
        dones.append(environment_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append(0.5 * (self.fire_update_probability) * environment_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(environment_reachable_transitions[i]['state'])
          fire_changed_state[0] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(environment_reachable_transitions[i]['reward'])
          dones.append(environment_reachable_transitions[i]['done'])
           
      for i in range(len(operator_reachable_transitions)):
        probs.append(0.5 * self.operator_play_probability * (1 - self.fire_update_probability) * operator_reachable_transitions[i]['prob'])
        new_states.append(operator_reachable_transitions[i]['state'])
        rewards.append(operator_reachable_transitions[i]['reward'])
        dones.append(operator_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append(0.5 * self.operator_play_probability * (self.fire_update_probability) * operator_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(operator_reachable_transitions[i]['state'])
          fire_changed_state[0] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(operator_reachable_transitions[i]['reward'])
          dones.append(operator_reachable_transitions[i]['done'])
          
      for update in fire_updates:
        probs.append(0.5 * (1 - self.operator_play_probability) * (self.fire_update_probability) * update['prob'])
        fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
        new_states.append(fire_changed_state)
        rewards.append(0)
        dones.append(False)
        
      if sum(probs) < 1:
        probs.append(1 - sum(probs))
        new_states.append(state)
        rewards.append(0)
        dones.append(False)
        
      assert sum(probs) == 1
      assert len(probs) == len(new_states) == len(rewards) == len(dones)
      return probs, new_states, rewards, dones
    
    
    if action == 1:
      # INTERRUPT OPERATOR
      # We find the reward and then make the operator call to transfer control
      probs = []
      new_states = []
      rewards = []
      dones = []
      reward = -25
      for flag in status_data:
        if flag == 3:
          reward = -40
          break
      operator_reachable_transitions = self.simulate_operator(fire_data, status_data, engagement_data)
      for i in range(len(operator_reachable_transitions)):
        probs.append((1 - self.fire_update_probability) * operator_reachable_transitions[i]['prob'])
        new_states.append(operator_reachable_transitions[i]['state'])
        rewards.append(reward + operator_reachable_transitions[i]['reward'])
        dones.append(operator_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append(self.fire_update_probability * operator_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(operator_reachable_transitions[i]['state'])
          fire_changed_state[0] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(operator_reachable_transitions[i]['reward'])
          dones.append(operator_reachable_transitions[i]['done'])
          
      if len(operator_reachable_transitions) == 0:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(reward)
          dones.append(False)
        
      if sum(probs) < 1:
        probs.append(1 - sum(probs))
        new_states.append(state)
        rewards.append(reward)
        dones.append(False)
        
      assert sum(probs) == 1
      assert len(probs) == len(new_states) == len(rewards) == len(dones)
      return probs, new_states, rewards, dones
    
    
    if action == 2:
      # DRONE 1 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[0] = 7
        engagement_data[0] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 3:
      # DRONE 2 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[1] = 7
        engagement_data[1] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 4:
      # DRONE 3 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[2] = 7
        engagement_data[2] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 5:
      # DRONE 1 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[0] = 4
        engagement_data[0] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 6:
      # DRONE 2 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[1] = 4
        engagement_data[1] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 7:
      # DRONE 1 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[2] = 4
        engagement_data[2] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 8:
      # DRONE 1 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[0] = 1
        if engagement_data[0][0] < 2:
          engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 9:
      # DRONE 2 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[1] = 1
        if engagement_data[1][0] < 2:
          engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 10:
      # DRONE 3 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        status_data[2] = 1
        if engagement_data[2][0] < 2:
          engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 11:
      # DRONE 3 SP NEGOTIATE
      # We find the reward and then change the state of the drone accordingly
      sp_busy = False
      for flag in status_data:
        if flag == 2:
          sp_busy = True
      if status_data[0] != 4 or sp_busy:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        status_data[0] = 2
        engagement_data[0][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 12:
      # DRONE 3 SP NEGOTIATE
      # We find the reward and then change the state of the drone accordingly
      sp_busy = False
      for flag in status_data:
        if flag == 2:
          sp_busy = True
      if status_data[1] != 4 or sp_busy:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        status_data[1] = 2
        engagement_data[1][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 13:
      # DRONE 3 SP NEGOTIATE
      # We find the reward and then change the state of the drone accordingly
      sp_busy = False
      for flag in status_data:
        if flag == 2:
          sp_busy = True
      if status_data[2] != 4 or sp_busy:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        status_data[2] = 2
        engagement_data[2][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([fire_data, status_data, engagement_data]))
        rewards.append(-10)
        dones.append(False)
      return probs, new_states, rewards, dones
  
  
  
  
  def simulate_operator(self, fire_data, status_data, engagement_data):
    '''
    Simulates moves made by the operator and modifies the state accordingly.
    Possible moves are, from highest priority to least:
      1. Take over and negotiate from a drone
      2. Issue Guide from a drone
      3. Issue Engage from a drone
      4. Issue SP negotiation from drone
      5. Issue SP negotiation from a drone or give a warning with equal probability
      6. Issue warning from a drone
    If none of this is possible, then we return an empty transition list. Operator does not make a move.
    '''
    
    
    operator_reachable_transitions = []
    
    
    '''
    1. If a drone is monitoring/ warning/ SP negotiating having already issued > 0 warnings and after one SP negotiation,
        the operator interrupts and takes over the communication. If the operator is already engaged, then nothing happens.
    '''
    num_possibilities = 0
    operator_busy = bool(status_data.count(3))
    possible_drones = []
    for i in range(3):
      if status_data[i] in [1,2,4] and engagement_data[i][0] > 0 and engagement_data[i][1] == 1:
        num_possibilities += 1
        possible_drones.append(i)
    
    if operator_busy:
      num_possibilities = 0
      
    if not operator_busy:
      for drone in possible_drones:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[drone] = 3
        transition = {}
        transition['prob'] = 1/num_possibilities
        transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition) 
      
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
      
        
    '''
    2. If a drone is ready to guide, issue a guide command.
    '''
    num_possibilities = 0
    possible_drones = []
    for i in range(3):
      if status_data[i] == 5:
        num_possibilities += 1
        possible_drones.append(i)
    
    for drone in possible_drones:
      temp_status_data = copy.deepcopy(status_data)
      temp_engagement_data = copy.deepcopy(engagement_data)
      temp_status_data[drone] = 7
      temp_engagement_data[drone]= [0,0]
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
    
    
    
    '''
    3. If a drone has found life, issue an engage command.
    '''
    num_possibilities = 0
    possible_drones = []
    for i in range(3):
      if status_data[i] == 6:
        num_possibilities += 1
        possible_drones.append(i)
    
    for drone in possible_drones:
      temp_status_data = copy.deepcopy(status_data)
      temp_engagement_data = copy.deepcopy(engagement_data)
      temp_status_data[drone] = 4
      temp_engagement_data[drone]= [0,0]
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
      
    
    
    '''
    4. If a drone has issued two warnings and is monitoring, start a SP negotiation if there are no other SP negotiations.
    '''
    num_possibilities = 0
    num_negotiations = status_data.count(2)
    possible_drones = []
    for i in range(3):
      if status_data[i] == 4 and engagement_data[i][0] == 2:
        num_possibilities += 1
        possible_drones.append(i)
    if num_negotiations> 0:
      num_possibilities = 0
    
    if num_negotiations == 0:
      for drone in possible_drones:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[drone] = 2
        temp_engagement_data[drone][1] = 1
        transition = {}
        transition['prob'] = 1/num_possibilities
        transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
    
    
    
    '''
    5. If a drone has issued one warning and is monitoring, issue a warning or start a SP negotiation with equal probability 
        if there are no other SP negotiations. If the SP is already negotiating, just warn.
    '''
    num_possibilities = 0
    num_negotiations = status_data.count(2)
    possible_drones = []
    for i in range(3):
      if status_data[i] == 4 and engagement_data[i][0] == 1:
        num_possibilities += 1
        possible_drones.append(i)
        if num_negotiations == 0:
          num_possibilities += 1
    
    for drone in possible_drones:
      temp_status_data = copy.deepcopy(status_data)
      temp_engagement_data = copy.deepcopy(engagement_data)
      temp_status_data[drone] = 1
      temp_engagement_data[drone][0] = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_negotiations == 0:
      for drone in possible_drones:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[drone] = 2
        temp_engagement_data[drone][1] = 1
        transition = {}
        transition['prob'] = 1/num_possibilities
        transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
    
  
  
    '''
    6. If a drone has not issued a warning yet and is monitoring, issue a warning.
    '''
    num_possibilities = 0
    possible_drones = []
    for i in range(3):
      if status_data[i] == 4 and engagement_data[i][0] == 0:
        num_possibilities += 1
        possible_drones.append(i)
    
    for drone in possible_drones:
      temp_status_data = copy.deepcopy(status_data)
      temp_engagement_data = copy.deepcopy(engagement_data)
      temp_status_data[drone] = 1
      temp_engagement_data[drone][0] = 1
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['state'] = self.encode([fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
      assert probs_sum == 1
      return operator_reachable_transitions
    
    
    assert operator_reachable_transitions == []
    return operator_reachable_transitions
      
    
      
  
  
  def simulate_environment(self, fire_data, status_data, engagement_data):
    '''
    Simulates events that occur in the environment and modifies the states accordingly.
    Does not update the location of the fire.
    Possible updates are:
      1. Civilian group is saved
      2. Civilian group has died
      3. Life has been found
      4. Group has been convinced. Guide call available.
      5. Warning has ended
      6. SP Negotiation has ended
      7. Operator negotiation has ended
    '''
    pass
    
    
    
    
    
  def update_fire_data(self, fire_data):
    temp_fire_data = copy.deepcopy(fire_data)
    fire_update_transitions = []
    
    if fire_data == [1,1,1,1]:
      transition = {}
      transition['prob'] = 1.0
      transition['fire_state'] = temp_fire_data
      fire_update_transitions.append(transition)
    elif fire_data.count(1) == 3:
      temp_fire_data = [1,1,1,1]
      num_possibilities = 1
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = temp_fire_data
      fire_update_transitions.append(transition)
    elif fire_data == [1,0,0,1]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,1,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,0,1]
      fire_update_transitions.append(transition)
    elif fire_data == [0,1,1,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,1,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,1,1]
      fire_update_transitions.append(transition)
    elif fire_data == [1,1,0,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,0,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,1,0]
      fire_update_transitions.append(transition)
    elif fire_data == [0,0,1,1]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,1,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,1,1]
      fire_update_transitions.append(transition)
    elif fire_data == [1,1,0,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,0,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,1,0]
      fire_update_transitions.append(transition)
    elif fire_data == [0,0,1,1]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,1,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,1,1]
      fire_update_transitions.append(transition)
    elif fire_data == [1,0,0,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,0,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,1,0]
      fire_update_transitions.append(transition)
    elif fire_data == [0,1,0,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,1,0,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,0,1]
      fire_update_transitions.append(transition)
    elif fire_data == [0,0,1,0]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,1,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,0,1,1]
      fire_update_transitions.append(transition)
    elif fire_data == [0,0,0,1]:
      num_possibilities = 2
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,0,1]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,0,1,1]
      fire_update_transitions.append(transition)
    elif fire_data == [0,0,0,0]:
      num_possibilities = 4
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [1,0,0,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,1,0,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,0,1,0]
      fire_update_transitions.append(transition)
      transition = {}
      transition['prob'] = 1/num_possibilities
      transition['fire_state'] = [0,0,0,1]
      fire_update_transitions.append(transition)
    
    return fire_update_transitions 
      
    
    
      
      
    
    
        
        
      
        
    
    
    
    
  