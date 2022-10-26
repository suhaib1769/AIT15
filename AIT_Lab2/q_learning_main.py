from turtle import clear
import simple_grid
from q_learning_skeleton import *
import gym


def act_loop(env, agent, num_episodes):
    outcomes = []
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()
        outcomes.append("Failure")

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report()
                print("state:", state)

            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)


            agent.process_experience(state, action, new_state, reward, done)
            state = new_state
            if done:
                if reward == 10:
                    outcomes[-1] = "Success"
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                agent.report()
                agent.report_policy()
                break

    env.close()
    print("final report:")
    agent.report_policy()
    return outcomes

def act_loop_after_training(env, agent):
    nb_success = 0
    for _ in range(100):
        state = env.reset()
        done = False

        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            action = agent.select_action(state, trained=True)
            new_state, reward, done, info = env.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            if(reward == 10):
                nb_success += 1
    env.close()
    return nb_success

EPSILON_DIFFERENCE = 0.000001

def value_iteration(env, gamma):
    num_a = env.action_space.n
    num_s = env.observation_space.n
    Q = np.zeros((num_s, num_a))
    while True:
        delta = 0
        Q_new = np.zeros((num_s, num_a))
        for s in range(num_s):
            for a in range(num_a):
                if(s != num_s - 1):
                    Q_new[s][a] = bellman(Q, env.P, s, a, gamma)
                    delta = max(delta, abs(Q[s][a] - Q_new[s][a]))
        Q = Q_new
        if delta < EPSILON_DIFFERENCE:
            return Q
        

def bellman(Q, P, s, a, gamma):
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

def policy_extraction(Q):
    policy = []
    for state in Q:
        policy.append(np.argmax(state))
    
    return policy

if __name__ == "__main__":
    ### CHOOSE ENVIRONMENT:
    env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    # env = simple_grid.DrunkenWalkEnv(map_name="theAlley")

    ### TRAINING THE AGENT AND EVALUATING BEHAVIOR AFTER TRAINING
    # num_a = env.action_space.n

    # if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
    #     num_o = env.observation_space.n
    # else:
    #     raise("Qtable only works for discrete observations")

    # discount = DEFAULT_DISCOUNT
    # ql = QLearner(num_o, num_a, discount) #<- QTable
    # outcomes = act_loop(env, ql, NUM_EPISODES)
    # success = act_loop_after_training(env, ql)
    # print(f"Success rate = {success}%")

    ### VALUE ITERATION
    # Q = value_iteration(env, 0.9)
    # print(Q)
    # policy = policy_extraction(Q)
    # print(policy)
