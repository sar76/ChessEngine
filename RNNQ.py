# this file is a basic reinforcement neural network that uses quality-learning 
# a Q-function estimates the future reward for taking an action
# Utilizing temporal distance learning, a method that allows for correction of previous states based on current

import numpy as np

class ReinforcementNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.001):
        # essentially we just need the number of neurons in each stage of the network
        # Anything that represents the network is included here as well (weight matrices):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1

        self.learning_rate = learning_rate
        # need to include this so that it can be accessed as an instance of the class
    
    def relu(self, x):
        return np.maximum(0, x)
        # returns x if x > 0
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
        
        # CONDITIONAL FUNCTION: if x > 0, then return 1 and if not return 0
    
    def forward(self, x):
        # as usual, forward prop stores the weighted sums of the non-input layers
        # we use a dot product here, look into vector * matrix dot products, as the math is a bit different
        self.hidden = self.relu(np.dot(x, self.weights1))
        # during forward prop, we multiply in the 'forward' direction, as matrix multiplication is not commutative
        self.output = self.relu(np.dot(self.hidden, self.weights2))

        return self.output
    
    def backward(self, x, y, output):
        # Here, x is the input given in the forward pass, y is the expected output, output is the output we calculated, and so we can solve for error
        self.output_error = y - output
        # sign in front of error tells you whether you increase or decrease weights
        # delta values give you the amount and direction to adjust the weights in
        self.output_delta = self.relu_derivative(output) * self.output_error

        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        # the reason we do this is so that we can calculate the 'error' at the hidden layer by distributing the error from the output
        # output delta tells us how much change we need and what neurons have the largest effect on this
        # so we multiply this by the weights between the hidden and output layer to give us a general idea of how much we need to adjust the weights by
        self.hidden_delta = self.hidden_error * self.relu_derivative(self.hidden)
        # Now using the deltas, we can adjust the weights:
        self.weights2 += self.learning_rate * np.dot(self.hidden.T.reshape(-1, 1), self.output_delta.reshape(1, -1))
        self.weights1 += self.learning_rate * np.dot(x.reshape(-1, 1), self.hidden_delta.reshape(1, -1))

class QLearningAgent:
    def __init__(self, state_size, action_size):
        # here, state_size is the dimensions of the state and action_size is the dimensions of the action that the agent can take
        self.state_size = state_size
        self.action_size = action_size
        # the paramter 4 just specifies the size of the hidden layer, or how many hidden neurons there are. 
        self.neural_network = ReinforcementNN(state_size, 32, action_size)
        self.gamma = 0.99
        # controls how much future rewards matter compared to immediate rewards
        # future rewards matter 0.95% of an immediate reward, as steps go on it follows the same pattern
        # for example, a reward of 100 in the next 3 steps would be worth 100 * (0.95^3) = 85.7
        self.epsilon = 1.0
        # controls explore vs exploit tradeoff, here, 1.0 means 100% random actions initially. 
        # decreases over time as the agent learns, explore = randomly choosing, exploit = choosing by exploiting knowledge
        self.epsilon_decay = 0.9995
        # rate at which exploration decreases
        # example: after 100 steps, rate of exploration is 1 * 0.995^100 = 0.61
        self.epsilon_min = 0.01 
        # sets a minimum so that agent is always exploring 1% of the time AT LEAST
        # prevents being stuck in a suboptimal state with no way out

    def get_action(self, state):
        # implements the epsilon-greedy strategy, which balances the exploration and exploitation in reinforcement learning
        if np.random.rand() <= self.epsilon:
            # if this random number between 0-1 is less than epsilon, we choose action randomly
            return np.random.randint(self.action_size)
            # return an action number between 0 and action_size - 1
            # we start from 0 hence we go to action_size - 1 and not action_size
        q_values = self.neural_network.forward(state)
        # state goes into input layer, propagates thru both hidden and output layer using weights1 and weights2
        # returns array of Q-values, one per action, [1.2, 1.8, 2.2] means that action 0,1,2 have respective rewards. 
        # picks the action with the highest reward
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        # paramters are all self-explanatory
        # "done" is just true or false and has to do with whether or not the episode is finished
        # An episode is one complete run for the agent from start to finish. 
        current_q_values = self.neural_network.forward(state)
        # gets the initial q values to start off with at input state
        # an example of what the state dimensions could like: [position, vekocity, angle]
        next_q_values = self.neural_network.forward(next_state)
        # do the same for the next q values
        # this is calculated to multiply by "gamma"
        if done:
            target = reward
        else:
            # if not done, include discounted rewards
            # pick the highest q value from next state, multiply by constant, and then add reward
            target = reward + self.gamma * np.max(next_q_values)
        
        target_q_values = current_q_values.copy()
        target_q_values[action] = target
        # updates q value of the action that was taken

        # now train the network thru backprop
        # y = target, as that is what is expected after evaluating the future action
        # output = current, what the network currently thinks is correct q values

        self.neural_network.backward(state, target_q_values, current_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # decay epsilon as it should be if it is still above the minimum


def train_agent(env, agent, episodes=1000):
    # this training agent will consist of the environment (think of it like the game)
    # the agent (the entity that makes the decisions), and the episodes, as defined earlier

    # Training loop for the Q-learning agent
    # env: OpenAI Gym-like environment
    # agent: QLearningAgent instance
    # episodes: Number of training episodes

    for episode in range(episodes):
        # the purpose of this loop is to reset the environment to its starting stage at the beginning of each episode
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # runs till the end of the episode
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            # what the step function does: 
                # Apply action to environment
                # Calculate next states
                # Determine reward
                # Check if episode is done
        
            total_reward += reward
            # iterate to next state and add to total reward for the episode
            agent.train(state, action, reward, next_state, done)
            state = next_state

        if episode % 100 == 0:
        # every 100 episodes, output the learning progress, between that is not needed as changes are minimal
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")




    

