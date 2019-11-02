# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:41:29 2019

@author: svaidyanath
"""

from gym.envs.toy_text import discrete
import numpy as np



class NewRSHumanSwarmEnv(discrete.DiscreteEnv):
  '''
  Class that will describe the environment of the Human Swarm Interaction. For detailed description of the problem, read the accompanying docs.
  The state variables are:
    1) operator busy time - gives an indication of for how long the operator will be busy
    2) status of group 
    3) dead time - time until the fire reaches the group
    4) guide details - what form of guide is required for thr group
    5) negotiation status - what kind of negotiation is required next
  There are 1440 states
  The action space is:
    There are 7 possible actions:
      0 - Wait
      1 - Interrupt Operator
      2 - Query for guide details
      3 - UAV Guide
      4 - Vehicle Guide
      5 - Warn
      6 - SPN
  '''
  
  
  def __init__(self, group_type='stubborn_couple', init_dead_time=None):
    
    num_states = 1440
    num_actions = 7
  
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
    
    self.group_type = group_type
    self.dead_time_update_prob = 0.2
    self.operator_busy_time_update_prob = 0.7
    self.guide_type = 0   
    self.convince_probs = {}
    if group_type == 'stubborn_couple':
      self.guide_type = 1  # self guide
      self.convince_probs['warn'] = 0.05
      self.convince_probs['spn'] = 0.15
      self.convince_probs['on'] = 1.0
    elif group_type == 'babysitter':
      self.guide_type = 2 # uav guide
      self.convince_probs['warn'] = 0.95
      self.convince_probs['spn'] = 0.99
      self.convince_probs['on'] = 1.0
    elif group_type == 'old_couple':
      self.guide_type = 3 # vehicle guide
      self.convince_probs['warn'] = 0.1
      self.convince_probs['spn'] = 0.95
      self.convince_probs['on'] = 1.0
    
    for op_busy in range(4):
      for status in range(6):
        for dead_time in range(5):
          for guide_type in range(4):
            for neg_status in range(3):
              state = self.encode([op_busy, status, dead_time, guide_type, neg_status])
              if init_dead_time is not None and status ==0 and dead_time == init_dead_time and guide_type == 0 and neg_status == 0:
                initial_state_distrib[state] += 1
              elif init_dead_time is None and status ==0 and dead_time >= 2 and guide_type == 0 and neg_status == 0:
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
    
    
  
  
  def encode(self, state):
    '''
    Function encodes the state into a number
    '''
    
    op_busy, status, dead_time, guide_type, neg_status = state
    
    i = op_busy
    i *= 6
    i += status
    i *= 5
    i += dead_time
    i *= 4
    i += guide_type
    i *= 3
    i += neg_status
    
    return i
  
  
  
  def decode(self, i):
    '''
    Function decodes the number into a state
    '''
    
    out = []
    out.append(i % 3)  #neg_status
    i = i // 3
    out.append(i % 4)  #guide_type
    i = i // 4
    out.append(i % 5) #dead_time
    i = i // 5
    out.append(i % 6) #status
    i = i // 6
    out.append(i) #op_busy
    i = i // 4
    assert i == 0
    
    out.reverse()
    
    return out
  
  
  
  def state_details(self, state):
    '''
    Gives explicit details about the state 
    Utility function useful for debugging
    '''
    op_busy, status, dead_time, guide_type, neg_status = self.decode(state)
    state_info = '['
    
    state_info += 'OP_BUSY={}, '.format(op_busy)
    
    if status == 0:
      state_info += 'STATUS=MONITORING, '
    elif status == 1:
      state_info += 'STATUS=WARNING, '
    elif status == 2:
      state_info += 'STATUS=SPN, '
    elif status == 3:
      state_info += 'STATUS=ON, '
    elif status == 4:
      state_info += 'STATUS=READY_TO_MOVE, '
    elif status == 5:
      state_info += 'STATUS=SAVED, '
      
    state_info += 'DEAD_TIME={}, '.format(dead_time)
      
    if guide_type == 0:
      state_info += 'GUIDE_TYPE=UNKNOWN, '
    elif guide_type == 1:
      state_info += 'GUIDE_TYPE=SELF, '
    elif guide_type == 2:
      state_info += 'GUIDE_TYPE=UAV, '
    elif guide_type == 3:
      state_info += 'GUIDE_TYPE=VEHICLE, '
      
    if neg_status == 0:
      state_info += 'NEGOTIATIONS=NONE.'
    elif neg_status == 1:
      state_info += 'NEGOTIATIONS=WARN(s).'
    elif neg_status == 2:
      state_info += 'NEGOTIATIONS=SPN(s).'
      
    return state_info + ']'
  
  
  
  def action_details(self, action):
    '''
    Gives details about the action
    Utility function useful for debugging
    '''
    
    if action == 0:
      return 'WAIT'
    if action == 1:
      return 'INTERRUPT-OPERATOR'
    if action == 2:
      return 'QUERY-FOR-GUIDE-DETAILS'
    if action == 3:
      return 'UAV GUIDE'
    if action == 4:
      return 'VEHICLE GUIDE'
    if action == 5:
      return 'WARN'
    if action == 6:
      return 'SP-NEGOTIATE'
  
  
  
  
  def __valid(self, state):
    '''
    Checks if the state is valid.
    A state is invalid iff:
      1) status is warning and negotiation status is 0
      2) status is SPN and negotiation status is 0 or 1
    '''
    
    op_busy, status, dead_time, guide_type, neg_status = self.decode(state)
    
    if status == 1 and neg_status == 0:
      return False
    if status == 2  and neg_status in [0,1]:
      return False
    
    return True
  
  
  
  def __terminal(self, state):
    '''
    Checks if the given state is a terminal state of the MDP
    A state is terminal iff:
      1) THe group is saved i.e. status = 5
      2) The fire has reached and the group has died i.e. dead_time = 0
    '''
    
    op_busy, status, dead_time, guide_type, neg_status = self.decode(state)
    
    if status == 5 or dead_time == 0:
      return True
    
    return False
  
  
  
  
  def __simulate(self, state, action):
    '''
    Simulates the action at the given state and returns the possible transitions
    '''
    
    op_busy, status, dead_time, guide_type, neg_status = self.decode(state)
    probs = []
    states = []
    rewards = []
    dones = []
    
    if action == 0:
      # Wait action. 
      # We just update the dead_time and op_busy
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      for transition in time_updated_transitions:
        probs.append(transition['prob'])
        states.append(transition['state'])
        rewards.append(transition['reward'])
        dones.append(transition['done'])
    
    if action == 1:
      # Interrupt Operator
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      
      for transition in time_updated_transitions:
        probs.append(transition['prob'])
        new_state = self.decode(transition['state'])
        new_state[0] = 3
        new_state[1] = 3
        new_state = self.encode(new_state)
        states.append(new_state)
        rewards.append(min(transition['reward'], -300 - (op_busy*500)))
        dones.append(transition['done'])
        
    
    if action == 2:
      # Query for guide details
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      if status != 4:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          states.append(transition['state'])
          rewards.append(min(transition['reward'], -3000))
          dones.append(transition['done'])
      
      else:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          new_state = self.decode(transition['state'])
          new_state[3] = self.guide_type
          done = False
          reward = 0
          if self.guide_type == 1:
            new_state[1] = 5
            reward = 5000
            done = True
          new_state = self.encode(new_state)
          states.append(new_state)
          if transition['reward'] == -5000:
            rewards.append(transition['reward'])
            dones.append(True)
          else:
            rewards.append(reward)
            dones.append(done)
          
    if action == 3:
      # UAV guide      
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      if status != 4:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          states.append(transition['state'])
          rewards.append(min(transition['reward'], -3000))
          dones.append(transition['done'])
      
      elif status == 4 and guide_type != 2:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          states.append(transition['state'])
          rewards.append(transition['reward'])
          dones.append(transition['done'])
          
      else:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          new_state = self.decode(transition['state'])
          new_state[1] = 5
          reward = 5000
          new_state = self.encode(new_state)
          states.append(new_state)
          if transition['reward'] == -5000:
            rewards.append(transition['reward'])
          else:
            rewards.append(reward)
          dones.append(True)
          
  
    if action == 4:
      # Vehicle guide
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      if status != 4:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          states.append(transition['state'])
          rewards.append(min(transition['reward'], -3000))
          dones.append(transition['done'])
      
      elif status ==4 and guide_type != 3:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          states.append(transition['state'])
          rewards.append(transition['reward'])
          dones.append(transition['done'])
      
      else:
        for transition in time_updated_transitions:
          probs.append(transition['prob'])
          new_state = self.decode(transition['state'])
          new_state[1] = 5
          reward = 5000
          new_state = self.encode(new_state)
          states.append(new_state)
          if transition['reward'] == -5000:
            rewards.append(transition['reward'])
          else:
            rewards.append(reward)
          dones.append(True)
          
          
    if action == 5:
      # Warn action
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      for transition in time_updated_transitions:
        probs.append(transition['prob'])
        new_state = self.decode(transition['state'])
        new_state[1] = 1
        new_state[4] = 1
        new_state = self.encode(new_state)
        states.append(new_state)
        rewards.append(transition['reward'])
        dones.append(transition['done'])
          
    
    if action == 6:
      # SPN action
      
      time_updated_transitions = self.__time_updates(op_busy, status, dead_time, guide_type, neg_status)
      for transition in time_updated_transitions:
        probs.append(transition['prob'])
        new_state = self.decode(transition['state'])
        new_state[1] = 2
        new_state[4] = 2
        new_state = self.encode(new_state)
        states.append(new_state)
        rewards.append(transition['reward'])
        dones.append(transition['done'])
          
    assert round(sum(probs), 15) == 1
    assert (np.array(probs) > 0).all()
    assert len(probs) == len(states) == len(rewards) == len(dones)
    return probs, states, rewards, dones
          
          
  
          
  def __time_updates(self, op_busy, status, dead_time, guide_type, neg_status):
    '''
    Updates the ending of a warning, an SPN, an ON, updates the dead_time and op_busy time as well
    '''
    time_updated_transitions = []
    new_op_busy = max(0, op_busy - 1)
    new_dead_time = max(0, dead_time - 1)
    done = False
    reward = 0
    if new_dead_time == 0:
      done = True
      reward = -5000
    
    if self.group_type == 'stubborn_couple':
      factor = 1.25
    if self.group_type == 'old_couple':
      factor = 1.5
    if self.group_type == 'babysitter':
      factor = 1.75
      
    warning_end_prob = 0.9
    spn_end_prob = (1/(1 + dead_time)) * (factor)
    on_end_prob = (1/(1 + dead_time)) * (factor)
    
    if status == 1:
      '''
      The group is being warned
      '''
      
      # if warning does not end and both times are updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * self.dead_time_update_prob * (1 - warning_end_prob)
      transition['state'] = self.encode([new_op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if warning does not end and only dead time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * (1 - warning_end_prob)
      transition['state'] = self.encode([op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if warning deos not end and only op_busy time is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * (1 - warning_end_prob)
      transition['state'] = self.encode([new_op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if warning does not end and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * (1 - warning_end_prob)
      transition['state'] = self.encode([op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and both times are updated
      transition = {}
      transition['prob'] = self.dead_time_update_prob * self.operator_busy_time_update_prob * warning_end_prob * self.convince_probs['warn']
      transition['state'] = self.encode([new_op_busy, 4, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and only dead time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * warning_end_prob * self.convince_probs['warn']
      transition['state'] = self.encode([op_busy, 4, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and only op_busy time is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * warning_end_prob * self.convince_probs['warn']
      transition['state'] = self.encode([new_op_busy, 4, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * warning_end_prob * self.convince_probs['warn']
      transition['state'] = self.encode([op_busy, 4, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and both times are updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * self.dead_time_update_prob * warning_end_prob * (1 - self.convince_probs['warn'])
      transition['state'] = self.encode([new_op_busy, 0, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and only dead_time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * warning_end_prob * (1 - self.convince_probs['warn'])
      transition['state'] = self.encode([op_busy, 0, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and only op_busy is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * warning_end_prob * (1 - self.convince_probs['warn'])
      transition['state'] = self.encode([new_op_busy, 0, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * warning_end_prob * (1 - self.convince_probs['warn'])
      transition['state'] = self.encode([op_busy, 0, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
    
    elif status == 2:
      '''
      the group is under SPN
      '''
      
      # if SPN does not end and both times are updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * self.dead_time_update_prob * (1 - spn_end_prob)
      transition['state'] = self.encode([new_op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if SPN does not end and only dead time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * (1 - spn_end_prob)
      transition['state'] = self.encode([op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if SPN deos not end and only op_busy time is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * (1 - spn_end_prob)
      transition['state'] = self.encode([new_op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if SPN does not end and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * (1 - spn_end_prob)
      transition['state'] = self.encode([op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and both times are updated
      transition = {}
      transition['prob'] = self.dead_time_update_prob * self.operator_busy_time_update_prob * spn_end_prob * self.convince_probs['spn']
      transition['state'] = self.encode([new_op_busy, 4, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and only dead time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * spn_end_prob * self.convince_probs['spn']
      transition['state'] = self.encode([op_busy, 4, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and only op_busy time is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * spn_end_prob * self.convince_probs['spn']
      transition['state'] = self.encode([new_op_busy, 4, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is convinced and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * spn_end_prob * self.convince_probs['spn']
      transition['state'] = self.encode([op_busy, 4, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and both times are updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * self.dead_time_update_prob * spn_end_prob * (1 - self.convince_probs['spn'])
      transition['state'] = self.encode([new_op_busy, 0, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and only dead_time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob * spn_end_prob * (1 - self.convince_probs['spn'])
      transition['state'] = self.encode([op_busy, 0, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and only op_busy is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) * spn_end_prob * (1 - self.convince_probs['spn'])
      transition['state'] = self.encode([new_op_busy, 0, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) * spn_end_prob * (1 - self.convince_probs['spn'])
      transition['state'] = self.encode([op_busy, 0, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      
    elif status == 3:
      '''
      the group is under ON
      Under ON, we do not update the operator busy time. The value goes to 0 when the ON ends
      '''
      
      
      # if ON does not end and dead time is updated
      transition = {}
      transition['prob'] = self.dead_time_update_prob * (1 - on_end_prob)
      transition['state'] = self.encode([op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
        
        
      # if ON does not end and dead time is not updated
      transition = {}
      transition['prob'] = (1 - self.dead_time_update_prob) * (1 - on_end_prob)
      transition['state'] = self.encode([op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
        
      
      
      # if group is convinced and dead time is updated
      transition = {}
      transition['prob'] = self.dead_time_update_prob * on_end_prob * self.convince_probs['on']
      transition['state'] = self.encode([0, 4, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      
      # if group is convinced and dead time is not updated
      transition = {}
      transition['prob'] = (1 - self.dead_time_update_prob) * on_end_prob * self.convince_probs['on']
      transition['state'] = self.encode([0, 4, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      
      # if group is not convinced and dead time is updated
      transition = {}
      transition['prob'] = self.dead_time_update_prob * on_end_prob * (1 - self.convince_probs['on'])
      transition['state'] = self.encode([0, 0, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # if group is not convinced and dead_time is not updated
      transition = {}
      transition['prob'] = (1 - self.dead_time_update_prob) * on_end_prob * (1 - self.convince_probs['on'])
      transition['state'] = self.encode([0, 0, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
        
    
    else:
      '''
      Other cases
      '''
      
      # both times are updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * self.dead_time_update_prob 
      transition['state'] = self.encode([new_op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # only dead time is updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * self.dead_time_update_prob 
      transition['state'] = self.encode([op_busy, status, new_dead_time, guide_type, neg_status])
      transition['reward'] = reward
      transition['done'] = done
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # only op_busy time is updated
      transition = {}
      transition['prob'] = self.operator_busy_time_update_prob * (1 - self.dead_time_update_prob) 
      transition['state'] = self.encode([new_op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
      
      # neither times are updated
      transition = {}
      transition['prob'] = (1 - self.operator_busy_time_update_prob) * (1 - self.dead_time_update_prob) 
      transition['state'] = self.encode([op_busy, status, dead_time, guide_type, neg_status])
      transition['reward'] = 0
      transition['done'] = False
      if transition['prob'] > 0:
        time_updated_transitions.append(transition)
        
    probs_sum = 0
    for transition in time_updated_transitions:
      assert transition['prob'] > 0
      probs_sum += transition['prob']
    assert round(probs_sum, 15) == 1
    
    return time_updated_transitions