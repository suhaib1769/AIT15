import simple_grid
import numpy as np

class valueIterator():

    def __init__(self, env, gamma = 0.9, epsilon_difference=0.000001) -> None:
        self.env = env
        self.gamma = gamma
        self.EPSILON_DIFFERENCE = epsilon_difference

    def value_iteration(self):
        num_a = self.env.action_space.n
        num_s = self.env.observation_space.n
        Q = np.zeros((num_s, num_a))
        while True:
            delta = 0
            Q_new = np.zeros((num_s, num_a))
            for s in range(num_s):
                for a in range(num_a):
                    if(s != num_s - 1):
                        Q_new[s][a] = self.bellman(Q, self.env.P, s, a, self.gamma)
                        delta = max(delta, abs(Q[s][a] - Q_new[s][a]))
            Q = Q_new
            if delta < self.EPSILON_DIFFERENCE:
                print("\nQ*:\n")
                return Q
            

    def bellman(self, Q, P, s, a, gamma):
        sum_states = 0
        for i in range(len(P[s][a])):
            next_state = P[s][a][i][1]
            reward = P[s][a][i][2]
            if(next_state == 4 or next_state == 8):
                reward += 0.2*simple_grid.BROKEN_LEG_PENALTY
            if(next_state == 12):
                reward += simple_grid.REWARD
            sum_states += P[s][a][i][0]*(reward + gamma*np.max(Q[next_state][:]))
        return sum_states

    def policy_extraction(self, Q):
        policy = []
        for state in Q:
            policy.append(np.argmax(state))
        
        print("\noptimal policy after value iteration:\n")
        print(policy)
