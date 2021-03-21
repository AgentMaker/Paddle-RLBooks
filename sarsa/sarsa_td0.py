import numpy as np
from cliffwalk.cliffwalk import GridWorld

def egreedy_policy(q_values, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])

def sarsa_td0(env,
              actions=['UP', 'DOWN', 'RIGHT', 'LEFT'],
              num_states=4*12,
              num_actions=4,
              epochs=500,
              render=True,
              exploration_rate=0.1,
              learning_rate=0.5,
              gamma=0.9):
    q = np.zeros((num_states, num_actions))

    reward_sum_list = []
    for i in range(epochs):
        state = env.reset()
        done = False
        reward_sum = 0
        action = egreedy_policy(q, state, exploration_rate)

        while not done:
            next_state, reward, done = env.step(action)
            reward_sum += reward
            next_action = egreedy_policy(q, next_state, exploration_rate)
            td_target = reward + gamma * q[next_state][next_action]
            td_error = td_target - q[state][action]
            q[state][action] += learning_rate * td_error

            state = next_state
            action = next_action

            if i % 100 == 0:
                env.render(q, action=actions[action], colorize_q=True)

        reward_sum_list.append(reward_sum)
        if i % 3 == 0:
            print('Average scores = ', np.mean(reward_sum_list))
            reward_sum_list = []

    return q

def train():
    env = GridWorld()
    q = sarsa_td0(env, render=False, learning_rate=0.5, gamma=0.99)

if __name__ == '__main__':
    train()