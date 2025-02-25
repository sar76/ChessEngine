"""
This implementation combines a custom CartPole environment with a neural Q-learning agent to solve the pole balancing problem. 
The system features: (1) A physics-accurate cart-pole simulation using Euler integration with 4D state space (position, velocity, 
angle, angular velocity) and discrete left/right actions; (2) A manually implemented 2-layer neural network (4x32x2) using ReLU hidden 
layers and linear Q-value outputs, trained through temporal difference learning; (3) ε-greedy exploration with exponential decay from 100% 
to 1% exploration over training; (4) Custom reward shaping that incentivizes both survival and center positioning; (5) Online learning without 
experience replay or target networks for educational clarity. The agent learns through episodic training to maximize discounted future rewards 
(γ=0.99) using pure NumPy operations, demonstrating core deep reinforcement learning concepts through explicit matrix calculations rather than 
deep learning frameworks.
"""

import numpy as np
from RNNQ import ReinforcementNN, QLearningAgent, train_agent

class CartPoleEnv:
   def __init__(self):
       self.gravity = 9.8
       self.cart_mass = 1.0
       self.pole_mass = 0.1
       self.pole_length = 0.5
       self.force_mag = 10.0
       self.tau = 0.02  # time step
       self.state_bounds = np.array([2.4, 10.0, 0.21, 10.0])
       self.state = None
       self.reset()
       
   def normalize_state(self, state):
       return np.clip(state/self.state_bounds, -1, 1)

   def reset(self):
       # Start with small random values
       self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
       # Allow wider angle range initially
       self.state[2] = np.random.uniform(low=-0.1, high=0.1)  
       return self.normalize_state(self.state)

   def step(self, action):
       x, x_dot, theta, theta_dot = self.state
       force = self.force_mag if action == 1 else -self.force_mag

       # Physics equations
       total_mass = self.cart_mass + self.pole_mass
       pole_mass_length = self.pole_mass * self.pole_length
       costheta = np.cos(theta)
       sintheta = np.sin(theta)

       temp = (force + pole_mass_length * theta_dot**2 * sintheta) / total_mass
       theta_acc = (self.gravity * sintheta - costheta * temp) / \
                  (self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / total_mass))
       x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass

       # Update state using Euler integration
       x += self.tau * x_dot
       x_dot += self.tau * x_acc
       theta += self.tau * theta_dot
       theta_dot += self.tau * theta_acc

       self.state = np.array([x, x_dot, theta, theta_dot])
       normalized_state = self.normalize_state(self.state)

       # Check failure conditions
       done = bool(
           x < -2.4 or x > 2.4 or  # Cart moved too far
           theta < -0.21 or theta > 0.21  # Pole angle too large
       )

       # Reward structure: base reward for staying up, bonus for being centered
       if done:
           reward = 0.0
       else:
           # 1.0 base reward plus up to 0.5 bonus for cart position
           reward = 1.0 + (1.0 - abs(x)/2.4) * 0.5

       return normalized_state, reward, done, {}

# Usage
env = CartPoleEnv()
agent = QLearningAgent(4, 2)  
train_agent(env, agent, episodes=1000)