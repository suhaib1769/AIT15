from turtle import clear
import simple_grid
import q_learning_value_iteration
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
#             action = agent.decaying_epsilon_greedy(state, t)
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
    print("\n-------------------------------------------------------------------------------------------")
    print("\nFINAL REPORT:\n")
    agent.report_policy()
    return outcomes

def act_loop_after_training(env, agent):
    nb_success = 0
    for _ in range(100):
        state = env.reset()
        done = False

        n = 0
        # Until the agent gets stuck or reaches the goal, keep training it
        while not done and n < 1000:
            action = agent.select_action(state, trained=True)
            new_state, reward, done, info = env.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            if(reward == 10):
                nb_success += 1
            n += 1

    env.close()
    print(f"\nSuccess rate = {nb_success}%\n")

if __name__ == "__main__":
    #######################
    ###   ENVIRONMENT   ###
    #######################
    # env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    env = simple_grid.DrunkenWalkEnv(map_name="theAlley")

    #######################
    ###    TRAINING     ###
    # #######################
    num_a = env.action_space.n

    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")

    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount) #<- QTable
    outcomes = act_loop(env, ql, NUM_EPISODES)
    # act_loop_after_training(env, ql)
    

    #######################
    ### VALUE ITERATION ###
    #######################
    value_iterator = q_learning_value_iteration.valueIterator(env)
    Q = value_iterator.value_iteration()
    print(Q)
    value_iterator.policy_extraction(Q)
