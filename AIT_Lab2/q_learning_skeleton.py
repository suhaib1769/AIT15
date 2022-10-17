import numpy as np

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1




class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.n_states = num_states
        self.n_actions = num_actions
        self.gamma = discount
        self.alpha = learning_rate

        self.Qtable = np.zeros((num_states,num_actions))
        self.policy = []


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass




    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        if done:
            # episode will terminate
            self.Qtable[state,action] = (1-self.alpha) * self.Qtable[state,action] + self.alpha*reward

        else:
            best_action = np.argmax(self.Qtable[next_state,:])
            self.Qtable[state,action] = (1-self.alpha) * self.Qtable[state,action] \
                                        + (self.alpha) * (reward + self.gamma * self.Qtable[next_state,best_action])



    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        if (np.random.random() < EPSILON):
            # exploration
            action = np.random.randint(self.n_actions)
            return action

        else:
            # exploitation
            bestactions = np.argwhere(self.Qtable[state,:] == np.max(self.Qtable[state,:])).flatten().tolist()
            chooseaction = np.random.randint(np.size(bestactions))
            return bestactions[chooseaction]


    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")

    def report_policy(self):
        """
        Find the current optimal policy
        """
        self.policy = []

        for i in range(0, self.n_states):
            optimal_action = np.argmax(self.Qtable[i])
            self.policy.append(optimal_action)

        print(np.array(self.policy))






        
