# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:46:26 2019

@author: svaidyanath
"""



from gym.envs.toy_text import discrete
import numpy as np
import copy
import pickle




class HumanSwarmEnv(discrete.DiscreteEnv):
  '''
  Class that will describe the environment of the Human Swarm Interaction. For detailed description of the problem, read the accompanying docs.
  The state variables are:
    For 3 drones, there are 28311552 states.
    npf - the number of people found/sighted by the drone
    nps - the number people already saved
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
  
  def __init__(self, num_groups_to_save=3, fire_update_probability=0.0625, operator_play_probability=0.3, warn_convince_prob=0.5, spn_convince_prob=0.75, on_convince_prob=1, warning_end_prob=0.9, spn_end_prob=0.8, on_end_prob=0.75, guide_end_prob=0.2, death_factor=0.05, data_ready=False):
    
    num_states = 28311552
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
      Done = True or False
    '''
    P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}
    
    if data_ready:
      with open('isd.pickle', 'rb') as fp:
        initial_state_distrib = pickle.load(fp)
      with open('P.pickle', 'rb') as fp:
        P = pickle.load(fp) 
      
    else:
      self.fire_update_probability = fire_update_probability  # in any state, when we move with any action to another state, there is a probability that the fire gets updated along with that action
      self.operator_play_probability = operator_play_probability   # we update the state with a random operator play that changes the state 'randomly' after a certain no. of 3 state ticks
      self.num_groups_to_save = num_groups_to_save  # the number of groups we need to save
      self.warn_convince_prob = warn_convince_prob  # probability with which guide call is available after a warning. Ease of convincing with a warning.
      self.spn_convince_prob = spn_convince_prob  # probability with which guide call is available after a SPN. Ease of convincing with a SPN.
      self.on_convince_prob = on_convince_prob  # probability with which guide call is available after a ON. Ease of convincing with the help of the operator.
      self.warning_end_prob = warning_end_prob  # probability with which a warning ends 
      self.spn_end_prob = spn_end_prob  # probability with which a SPN ends 
      self.on_end_prob = on_end_prob  # probability with which a ON ends 
      self.guide_end_prob = guide_end_prob  # probability with which a guide ends
      self.death_factor = death_factor  #factor by which we multiply to find probability of someone dying
    
    
      for npf in range(4):
        for nps in range(4):
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
                                    state = self.encode([npf, nps, fire_data, status_data, engagement_data])
                                    if npf == nps == s1 == s2 == s3 == e1w == e1n == e2w == e2n == e3w == e3n == 0:
                                      if sum(fire_data) == 1:
                                        initial_state_distrib[state] += 1
                                    for action in range(num_actions):
                                      if self.__valid(state):
                                        probs, new_states, rewards, dones = self.__simulate(state, action)
                                        probs, new_states, rewards, dones = self.__kill_updates(probs, new_states, rewards, dones)
                                        for i in range(len(probs)):
                                          if probs[i] > 0:
                                            new_state = self.decode(new_states[i])
                                            new_npf, _, _, new_status_data, _ = new_state
                                            num_engaged_drones = 3 - new_status_data.count(0)
                                            if new_npf - num_engaged_drones == 3:
                                              dones[i] = True
                                            P[state][action].append((probs[i], new_states[i], rewards[i], dones[i]))
                                      else:
                                        P[state][action].append(1.0, state, 0, False)
                                      print(state, action)
                                    print('STATE {} COMPLETED'.format(state))
                                
      initial_state_distrib /= initial_state_distrib.sum()
      
      with open('isd.pickle', 'wb') as fp:
        pickle.dump(initial_state_distrib, fp)
      with open('P.pickle', 'wb') as fp:
        pickle.dump(P, fp)
    
    discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)
  
  
  
    
    
  def encode(self, state):
    '''
    Function encodes the state into a number
    npf = number of people found
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
    npf, nps, fire_data, status_data, engagement_data = state
    tl, tr, bl, br = fire_data
    s1, s2, s3 = status_data
    e1, e2, e3 = engagement_data
    e1w, e1n = e1
    e2w, e2n = e2
    e3w, e3n = e3
    i = npf
    i *= 4
    i += nps
    i *= 2
    i += tl
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
    npf = number of people found
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
    out.append(i % 2)  #tl
    i = i // 2
    out.append(i % 4)  #nps
    i = i // 4
    out.append(i)  #npf
    i = i // 4
    assert i == 0
    out.reverse()
    npf = out[0]
    nps = out[1]
    fire_data = [out[2], out[3], out[4], out[5]]
    status_data = [out[6], out[7], out[8]]
    engagement_data = [[out[9], out[10]], [out[11], out[12]], [out[13], out[14]]]
    return [npf, nps, fire_data, status_data, engagement_data]
  
  
  
  
  
  def __kill_updates(self, probs, new_states, rewards, dones):
    '''
    At the end of every action simulation, updates the states and the probabilities by adding probabilities of killing 
    a civilian group as well.
    '''
    
    updated_probs = copy.deepcopy(probs)
    updated_new_states = copy.deepcopy(new_states)
    updated_rewards = copy.deepcopy(rewards)
    updated_dones = copy.deepcopy(dones)
    
    for i in range(len(probs)):
      npf, nps, fire_data, status_data, engagement_data = self.decode(updated_new_states[i])
      kill_prob = (sum(fire_data)/4) * self.death_factor
      num_guiding_drones = status_data.count(7)  # not killable
      num_killable_groups = self.num_groups_to_save - nps - num_guiding_drones
      killable_drones = [i for i in range(3) if status_data[i] != 7 and status_data[i] != 0]
      old_prob = updated_probs[i]
      updated_probs[i] *= 1 - kill_prob
      for drone in killable_drones:
        temp_status_data = copy.deepcopy(status_data)
        temp_status_data[drone] = 0
        updated_probs.append((1/num_killable_groups) * kill_prob * old_prob)
        updated_new_states.append(self.encode([npf, nps, fire_data, temp_status_data, engagement_data]))
        updated_rewards.append(-500000 + updated_rewards[i])
        updated_dones.append(False)
      updated_probs.append((1 - len(killable_drones)/num_killable_groups) * kill_prob * old_prob)
      updated_new_states.append(self.encode([npf + 1, nps, fire_data, status_data, engagement_data]))   
      updated_rewards.append(-500000 + updated_rewards[i])
      updated_dones.append(False)
      
    assert round(sum(updated_probs), 15) == 1
    assert (np.array(updated_probs) >= 0).all()
    assert len(updated_probs) == len(updated_new_states) == len(updated_rewards) == len(updated_dones)
    return updated_probs, updated_new_states, updated_rewards, updated_dones
    
    
  
  
  
  
  def __valid(self, state):
    '''
    Returns True or False depending on whether the state is valid or not
    A state is invalid if any of the following conditions hold:
      1) npf < nps + engaged drones (status != 0)
      2) num_groups_to_save < nps + engaged_drones (status != 0)
      3) Non zero engagement data for drone in status 0, 6, 7
      4) No fire
      5) Drone warning/SPN but engagement data does not reflect it.
    '''
    npf, nps, fire_data, status_data, engagement_data = self.decode(state)
    num_engaged_drones = 3 - status_data.count(0)
    
    if npf < nps + num_engaged_drones:
      return False
    if self.num_groups_to_save < nps + num_engaged_drones:
      return False
    for drone in range(3):
      if status_data[drone] in [0, 6, 7] and sum(engagement_data[drone]) > 0:
        return False
      if status_data[drone] == 1 and engagement_data[drone][0] == 0:
        return False
      if status_data[drone] == 2 and engagement_data[drone][1] == 0:
        return False
    if sum(fire_data) == 0:
      return False
    
    return True
    
    
  
  
  def __simulate(self, state, action):
    npf, nps, fire_data, status_data, engagement_data = self.decode(state)
    
    fire_updates = self.__update_fire_data(fire_data)
    probs = []
    new_states = []
    rewards = []
    dones = []
    
    if action == 0:
      # NOP ACTION
      # We have plays from the environment and/or from the operator
      environment_reachable_transitions = self.__simulate_environment(npf, nps, fire_data, status_data, engagement_data)
      operator_reachable_transitions = self.__simulate_operator(npf, nps, fire_data, status_data, engagement_data)
      for i in range(len(environment_reachable_transitions)):
        if environment_reachable_transitions[i]['prob'] == 0:
          continue
        probs.append((1 - self.operator_play_probability) * (1 - self.fire_update_probability) * environment_reachable_transitions[i]['prob'])
        new_states.append(environment_reachable_transitions[i]['state'])
        rewards.append(environment_reachable_transitions[i]['reward'])
        dones.append(environment_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append((1 - self.operator_play_probability) * (self.fire_update_probability) * environment_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(environment_reachable_transitions[i]['state'])
          fire_changed_state[2] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(environment_reachable_transitions[i]['reward'])
          dones.append(environment_reachable_transitions[i]['done'])
           
      for i in range(len(operator_reachable_transitions)):
        probs.append(self.operator_play_probability * (1 - self.fire_update_probability) * operator_reachable_transitions[i]['prob'])
        new_states.append(operator_reachable_transitions[i]['state'])
        rewards.append(operator_reachable_transitions[i]['reward'])
        dones.append(operator_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append(self.operator_play_probability * (self.fire_update_probability) * operator_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(operator_reachable_transitions[i]['state'])
          fire_changed_state[2] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(operator_reachable_transitions[i]['reward'])
          dones.append(operator_reachable_transitions[i]['done'])
      
      if sum(probs) < 1:
        assert (operator_reachable_transitions == []) or (environment_reachable_transitions == [])
        remainder = 1 - sum(probs)
        for update in fire_updates:
          probs.append(remainder * (self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(0)
          dones.append(False)
        probs.append(remainder * (1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(0)
        dones.append(False)
          
      assert round(sum(probs), 15) == 1
      assert (np.array(probs) >= 0).all()
      assert len(probs) == len(new_states) == len(rewards) == len(dones)
      return probs, new_states, rewards, dones
    
    
    if action == 1:
      # INTERRUPT OPERATOR
      # We find the reward and then make the operator call to transfer control
      reward = -25
      for flag in status_data:
        if flag == 3:
          reward = -40
          break
      operator_reachable_transitions = self.__simulate_operator(npf, nps, fire_data, status_data, engagement_data)
      for i in range(len(operator_reachable_transitions)):
        probs.append((1 - self.fire_update_probability) * operator_reachable_transitions[i]['prob'])
        new_states.append(operator_reachable_transitions[i]['state'])
        rewards.append(reward + operator_reachable_transitions[i]['reward'])
        dones.append(operator_reachable_transitions[i]['done'])
        for update in fire_updates:
          probs.append(self.fire_update_probability * operator_reachable_transitions[i]['prob'] * update['prob'])
          fire_changed_state = self.decode(operator_reachable_transitions[i]['state'])
          fire_changed_state[2] = update['fire_state']
          fire_changed_state = self.encode(fire_changed_state)
          new_states.append(fire_changed_state)
          rewards.append(operator_reachable_transitions[i]['reward'])
          dones.append(operator_reachable_transitions[i]['done'])
          
      if len(operator_reachable_transitions) == 0:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(reward)
          dones.append(False)
        
      if sum(probs) < 1:
        assert operator_reachable_transitions == []
        remainder = 1 - sum(probs)
        for update in fire_updates:
          probs.append(remainder * (self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(0)
          dones.append(False)
        probs.append(remainder * (1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(0)
        dones.append(False)
        
      assert round(sum(probs), 15) == 1
      assert (np.array(probs) >= 0).all()
      assert len(probs) == len(new_states) == len(rewards) == len(dones)
      return probs, new_states, rewards, dones
    
    
    if action == 2:
      # DRONE 1 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[0] = 7
        temp_engagement_data[0] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 3:
      # DRONE 2 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[1] = 7
        temp_engagement_data[1] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 4:
      # DRONE 3 GUIDE
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 5:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[2] = 7
        temp_engagement_data[2] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 5:
      # DRONE 1 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[0] = 4
        temp_engagement_data[0] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 6:
      # DRONE 2 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        status_data[1] = 4
        engagement_data[1] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 7:
      # DRONE 3 ENGAGE
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 6:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        status_data[2] = 4
        engagement_data[2] = [0,0]
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(+10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(+10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 8:
      # DRONE 1 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[0] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[0] = 1
        if engagement_data[0][0] < 2:
          temp_engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 9:
      # DRONE 2 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[1] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[1] = 1
        if engagement_data[1][0] < 2:
          temp_engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 10:
      # DRONE 3 WARN
      # We find the reward and then change the state of the drone accordingly
      if status_data[2] != 4:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-10)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[2] = 1
        if engagement_data[2][0] < 2:
          temp_engagement_data += 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-3)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(-3)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 11:
      # DRONE 1 SP NEGOTIATE
      # We find the reward and then change the state of the drone accordingly
      sp_busy = False
      for flag in status_data:
        if flag == 2:
          sp_busy = True
      if status_data[0] != 4 or sp_busy:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[0] = 2
        temp_engagement_data[0][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(-10)
        dones.append(False)
      return probs, new_states, rewards, dones
    
    
    if action == 12:
      # DRONE 2 SP NEGOTIATE
      # We find the reward and then change the state of the drone accordingly
      sp_busy = False
      for flag in status_data:
        if flag == 2:
          sp_busy = True
      if status_data[1] != 4 or sp_busy:
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[1] = 2
        temp_engagement_data[1][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
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
          fire_changed_state = self.encode([npf, nps, update['fire_state'], status_data, engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-20)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(state)
        rewards.append(-20)
        dones.append(False)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_engagement_data = copy.deepcopy(engagement_data)
        temp_status_data[2] = 2
        temp_engagement_data[2][1] = 1
        for update in fire_updates:
          probs.append((self.fire_update_probability) * update['prob'])
          fire_changed_state = self.encode([npf, nps, update['fire_state'], temp_status_data, temp_engagement_data])
          new_states.append(fire_changed_state)
          rewards.append(-10)
          dones.append(False)
        probs.append((1 - self.fire_update_probability))
        new_states.append(self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data]))
        rewards.append(-10)
        dones.append(False)
      return probs, new_states, rewards, dones
  
  
  
  
  def __simulate_operator(self, npf, nps, fire_data, status_data, engagement_data):
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
        transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition) 
      
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
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
      transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
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
      transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
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
        transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
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
      transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
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
        transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
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
      transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, temp_engagement_data])
      transition['reward'] = 0
      transition['done'] = False
      operator_reachable_transitions.append(transition)
    
    if num_possibilities > 0:
      probs_sum = 0
      for transition in operator_reachable_transitions:
        probs_sum += transition['prob']
        assert transition['prob'] >= 0
      assert probs_sum == 1
      return operator_reachable_transitions
    
    
    assert operator_reachable_transitions == []
    return operator_reachable_transitions
      
    
      
  
  
  def __simulate_environment(self, npf, nps, fire_data, status_data, engagement_data):
    '''
    Simulates events that occur in the environment and modifies the states accordingly.
    Possible updates are:
      1. Civilian group is saved i.e. guide ends (only from status 7)
      2. Life has been found (only from status 0)
      3. Group has been convinced. Guide call available.  (from status 1,2,3)
      4. Warning has ended (only from status 1)
      5. SP Negotiation has ended (only from status 2)
      6. Operator negotiation has ended (only from status 3)
    '''
    environment_reachable_transitions = []
    
    guiding_drones = [i for i in range(3) if status_data[i] == 7]
    searching_drones = [i for i in range(3) if status_data[i] == 0]
    warning_drones = [i for i in range(3) if status_data[i] == 1]
    spn_drones = [i for i in range(3) if status_data[i] == 2]
    on_drones = [i for i in range(3) if status_data[i] == 3]
    if len(guiding_drones) > 0:
      for drone in guiding_drones:
        temp_status_data = copy.deepcopy(status_data)
        temp_status_data[drone] = 0
        if len(searching_drones) > 0:
          branch_point_one_transitions = self.__branch_point_one(npf, nps, fire_data, temp_status_data, engagement_data)
          for transition in branch_point_one_transitions:
            transition['prob'] *= self.guide_end_prob * 1/len(guiding_drones)
            transition['reward'] += 5000
            environment_reachable_transitions.append(transition)
          branch_point_one_transitions = self.__branch_point_one(npf, nps, fire_data, status_data, engagement_data)
          for transition in branch_point_one_transitions:
            transition['prob'] *= (1 - self.guide_end_prob) * 1/len(guiding_drones)
            environment_reachable_transitions.append(transition)
        else:
          if len(warning_drones) > 0:
            branch_point_two_transitions = self.__branch_point_two(npf, nps, fire_data, temp_status_data, engagement_data)
            for transition in branch_point_two_transitions:
              transition['prob'] *= self.guide_end_prob * 1/len(guiding_drones)
              transition['reward'] += 5000
              environment_reachable_transitions.append(transition)
            branch_point_two_transitions = self.__branch_point_two(npf, nps, fire_data, status_data, engagement_data)
            for transition in branch_point_two_transitions:
              transition['prob'] *= (1 - self.guide_end_prob) * 1/len(guiding_drones)
              environment_reachable_transitions.append(transition)
          else:
            if len(spn_drones) > 0:
              branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, temp_status_data, engagement_data)
              for transition in branch_point_three_transitions:
                transition['prob'] *= self.guide_end_prob * 1/len(guiding_drones)
                transition['reward'] += 5000
                environment_reachable_transitions.append(transition)
              branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, status_data, engagement_data)
              for transition in branch_point_three_transitions:
                transition['prob'] *= (1 - self.guide_end_prob) * 1/len(guiding_drones)
                environment_reachable_transitions.append(transition)
            else:
              if len(on_drones) > 0:
                branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data)
                for transition in branch_point_four_transitions:
                  transition['prob'] *= self.guide_end_prob * 1/len(guiding_drones)
                  transition['reward'] += 5000
                  environment_reachable_transitions.append(transition)
                branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, status_data, engagement_data)
                for transition in branch_point_four_transitions:
                  transition['prob'] *= (1 - self.guide_end_prob) * 1/len(guiding_drones)
                  environment_reachable_transitions.append(transition)
              else:
                transition = {}
                transition['prob'] = 1/len(guiding_drones) * self.guide_end_prob
                transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
                transition['reward'] = +5000
                transition['done'] = False
                environment_reachable_transitions.append(transition)
                transition = {}
                transition['prob'] = 1/len(guiding_drones) * (1 - self.guide_end_prob)
                transition['state'] = self.encode([npf, nps, fire_data, status_data, engagement_data])
                transition['reward'] = 0
                transition['done'] = False
                environment_reachable_transitions.append(transition)
        
    else:
      if len(searching_drones) > 0:
          environment_reachable_transitions.extend(self.__branch_point_one(npf, nps, fire_data, status_data, engagement_data))
      else:
        if len(warning_drones) > 0:
          environment_reachable_transitions.extend(self.__branch_point_two(npf, nps, fire_data, status_data, engagement_data))
        else:
          if len(spn_drones) > 0:
            environment_reachable_transitions.extend(self.__branch_point_three(npf, nps, fire_data, status_data, engagement_data))
          else:
            if len(on_drones) > 0:
              environment_reachable_transitions.extend(self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data))
      
    probs_sum = 0
    for transition in environment_reachable_transitions:
      probs_sum += transition['prob']
      assert transition['prob'] >= 0
    probs_sum = round(probs_sum, 15)
    assert (probs_sum == 1) or (environment_reachable_transitions == [])
    return environment_reachable_transitions
    
    
    
  def __branch_point_one(self, npf, nps, fire_data, status_data, engagement_data):
    '''
    Utility function to make probability visualisation and calculation easier
    Called if at least one group is searching
    '''
    
    branch_point_one_transitions = []
    
    prob_lifefound = (self.num_groups_to_save - npf)/ self.num_groups_to_save
    searching_drones = [i for i in range(3) if status_data[i] == 0]
    warning_drones = [i for i in range(3) if status_data[i] == 1]
    spn_drones = spn_drones = [i for i in range(3) if status_data[i] == 2]
    on_drones = on_drones = [i for i in range(3) if status_data[i] == 3]
    for drone in searching_drones:
      temp_status_data = copy.deepcopy(status_data)
      temp_status_data[drone] = 6
      if len(warning_drones) > 0:
        branch_point_two_transitions = self.__branch_point_two(npf + 1, nps, fire_data, temp_status_data, engagement_data)
        for transition in branch_point_two_transitions:
           transition['prob'] *= 1/len(searching_drones) * prob_lifefound
           branch_point_one_transitions.append(transition)
        branch_point_two_transitions = self.__branch_point_two(npf, nps, fire_data, status_data, engagement_data)
        for transition in branch_point_two_transitions:
          transition['prob'] *= 1/len(searching_drones) * (1 - prob_lifefound) 
          branch_point_one_transitions.append(transition)
      else:
        if len(spn_drones) > 0:
          branch_point_three_transitions = self.__branch_point_three(npf + 1, nps, fire_data, temp_status_data, engagement_data)
          for transition in branch_point_three_transitions:
            transition['prob'] *= 1/len(searching_drones) * prob_lifefound
            branch_point_one_transitions.append(transition)
          branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, status_data, engagement_data)
          for transition in branch_point_three_transitions:
            transition['prob'] *=  1/len(searching_drones) * (1 - prob_lifefound)
            branch_point_one_transitions.append(transition)
        else:
          if len(on_drones) > 0:
            branch_point_four_transitions = self.__branch_point_four(npf + 1, nps, fire_data, temp_status_data, engagement_data)
            for transition in branch_point_four_transitions:
              transition['prob'] *= 1/len(searching_drones) * prob_lifefound 
              branch_point_one_transitions.append(transition)
            branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, status_data, engagement_data)
            for transition in branch_point_four_transitions:
              transition['prob'] *=  1/len(searching_drones) * (1 - prob_lifefound) 
              branch_point_one_transitions.append(transition)
          else:
            transition = {}
            transition['prob'] = 1/len(searching_drones) * prob_lifefound
            transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
            transition['reward'] = 0
            transition['done'] = False
            branch_point_one_transitions.append(transition)
            transition = {}
            transition['prob'] = 1/len(searching_drones) * (1 - prob_lifefound)
            transition['state'] = self.encode([npf, nps, fire_data, status_data, engagement_data])
            transition['reward'] = 0
            transition['done'] = False
            branch_point_one_transitions.append(transition)
    probs_sum = 0
    for transition in branch_point_one_transitions:
      probs_sum += transition['prob']
      assert transition['prob'] >= 0
    probs_sum = round(probs_sum, 15)
    assert probs_sum == 1
    return branch_point_one_transitions
            
      
      
      
  def __branch_point_two(self, npf, nps, fire_data, status_data, engagement_data):
    '''
    Utility function to make probability visualisation and calculation easier
    Called only if there is some drone in Warning phase
    '''
    branch_point_two_transitions = []
    warning_drones = [i for i in range(3) if status_data[i] == 1]
    spn_drones = [i for i in range(3) if status_data[i] == 2]
    on_drones = [i for i in range(3) if status_data[i] == 3]
    
    for drone in warning_drones:
      if len(spn_drones) > 0:
        temp_status_data = copy.deepcopy(status_data)
        temp_status_data[drone] = 4
        branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, temp_status_data, engagement_data)
        for transition in branch_point_three_transitions:
          transition['prob'] *= 1/len(warning_drones) * self.warning_end_prob * (1 - self.warn_convince_prob)
          branch_point_two_transitions.append(transition)
        temp_status_data[drone] = 5
        branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, temp_status_data, engagement_data)
        for transition in branch_point_three_transitions:
          transition['prob'] *= 1/len(warning_drones) * self.warning_end_prob * self.warn_convince_prob
          branch_point_two_transitions.append(transition)
        branch_point_three_transitions = self.__branch_point_three(npf, nps, fire_data, status_data, engagement_data)
        for transition in branch_point_three_transitions:
          transition['prob'] *= 1/len(warning_drones) * (1 - self.warning_end_prob)
          branch_point_two_transitions.append(transition)
      else:
        if len(on_drones) > 0:
          temp_status_data = copy.deepcopy(status_data)
          temp_status_data[drone] = 4
          branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data)
          for transition in branch_point_four_transitions:
            transition['prob'] *= 1/len(warning_drones) * self.warning_end_prob * (1 - self.warn_convince_prob)
            branch_point_two_transitions.append(transition)
          temp_status_data[drone] = 5
          branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data)
          for transition in branch_point_four_transitions:
            transition['prob'] *= 1/len(warning_drones) * self.warning_end_prob * self.warn_convince_prob
            branch_point_two_transitions.append(transition)
          branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, status_data, engagement_data)
          for transition in branch_point_four_transitions:
            transition['prob'] *= 1/len(warning_drones) * (1 - self.warning_end_prob)
            branch_point_two_transitions.append(transition)
        else:
          temp_status_data = copy.deepcopy(status_data)
          temp_status_data[drone] = 4
          transition = {}
          transition['prob'] = self.warning_end_prob * (1 - self.warn_convince_prob) * 1/len(warning_drones)
          transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
          transition['reward'] = 0
          transition['done'] = False
          branch_point_two_transitions.append(transition)
          temp_status_data[drone] = 5
          transition = {}
          transition['prob'] = self.warning_end_prob * self.warn_convince_prob * 1/len(warning_drones)
          transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
          transition['reward'] = 0
          transition['done'] = False
          branch_point_two_transitions.append(transition)
        transition = {}
        transition['prob'] = 1/len(warning_drones) * (1 - self.warning_end_prob)
        transition['state'] = self.encode([npf, nps, fire_data, status_data, engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        branch_point_two_transitions.append(transition)
    probs_sum = 0
    for transition in branch_point_two_transitions:
      probs_sum += transition['prob']
      assert transition['prob'] >= 0
    probs_sum = round(probs_sum, 15)
    assert probs_sum == 1
    return branch_point_two_transitions 
    
    
  
  def __branch_point_three(self, npf, nps, fire_data, status_data, engagement_data):
    '''
    Utility function to make probability visualisation and calculation easier
    Called only if there is some drone in SPN phase
    '''
    branch_point_three_transitions = []
    spn_drones = [i for i in range(3) if status_data[i] == 2]
    on_drones = [i for i in range(3) if status_data[i] == 3]  # there can be at max only one ON drone
    
    for drone in spn_drones:
      if len(on_drones) > 0:
        temp_status_data = copy.deepcopy(status_data)
        temp_status_data[drone] = 4
        branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data)
        for transition in branch_point_four_transitions:
          transition['prob'] *= 1/len(spn_drones) * self.spn_end_prob * (1 - self.spn_convince_prob)
          branch_point_three_transitions.append(transition)
        temp_status_data[drone] = 5
        branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, temp_status_data, engagement_data)
        for transition in branch_point_four_transitions:
          transition['prob'] *= 1/len(spn_drones) * self.spn_end_prob * self.spn_convince_prob
          branch_point_three_transitions.append(transition)
        branch_point_four_transitions = self.__branch_point_four(npf, nps, fire_data, status_data, engagement_data)
        for transition in branch_point_four_transitions:
          transition['prob'] *= 1/len(spn_drones) * (1 - self.spn_end_prob)
          branch_point_three_transitions.append(transition)
      else:
        temp_status_data = copy.deepcopy(status_data)
        temp_status_data[drone] = 4
        transition = {}
        transition['prob'] = self.spn_end_prob * (1 - self.spn_convince_prob) * 1/len(spn_drones)
        transition['prob'] = transition['prob']
        transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        branch_point_three_transitions.append(transition)
        temp_status_data[drone] = 5
        transition = {}
        transition['prob'] = self.spn_end_prob * self.spn_convince_prob * 1/len(spn_drones)
        transition['prob'] = transition['prob']
        transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        branch_point_three_transitions.append(transition)
        transition = {}
        transition['prob'] = 1/len(spn_drones) * (1 - self.spn_end_prob)
        transition['prob'] = transition['prob']
        transition['state'] = self.encode([npf, nps, fire_data, status_data, engagement_data])
        transition['reward'] = 0
        transition['done'] = False
        branch_point_three_transitions.append(transition)
    probs_sum = 0
    for transition in branch_point_three_transitions:
      probs_sum += transition['prob']
      assert transition['prob'] >= 0
    probs_sum = round(probs_sum, 15)
    assert probs_sum == 1
    return branch_point_three_transitions
    
    
    
  def __branch_point_four(self, npf, nps, fire_data, status_data, engagement_data):
    '''
    Utility function to make probability visualisation and calculation easier
    Called only if there is some drone in ON phase
    '''
    branch_point_four_transitions = []
    
    on_drone = status_data.index(3)   # there can only be one ON drone at a time
    
    temp_status_data = copy.deepcopy(status_data)
    temp_status_data[on_drone] = 4
    transition = {}
    transition['prob'] = self.on_end_prob * (1 - self.on_convince_prob)
    transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
    transition['reward'] = 0
    transition['done'] = False
    branch_point_four_transitions.append(transition)
    temp_status_data[on_drone] = 5
    transition = {}
    transition['prob'] = self.on_end_prob * self.on_convince_prob
    transition['state'] = self.encode([npf, nps, fire_data, temp_status_data, engagement_data])
    transition['reward'] = 0
    transition['done'] = False
    branch_point_four_transitions.append(transition)
    transition = {}
    transition['prob'] = 1 - self.on_end_prob
    transition['state'] = self.encode([npf, nps, fire_data, status_data, engagement_data])
    transition['reward'] = 0
    transition['done'] = False
    branch_point_four_transitions.append(transition)
    probs_sum = 0
    for transition in branch_point_four_transitions:
      probs_sum += transition['prob']
      assert transition['prob'] >= 0
    probs_sum = round(probs_sum, 15)
    assert probs_sum == 1
    return branch_point_four_transitions
    
    
        
    
    
  def __update_fire_data(self, fire_data):
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
      
    
    
      
      
    
    
        
        
      
        
    
    
    
    
  