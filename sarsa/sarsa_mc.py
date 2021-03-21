import numpy as np
from cliffwalk.cliffwalk import GridWorld

def egreedy_policy(q_values, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])

def sarsa_mc(env,
              actions=['UP', 'DOWN', 'RIGHT', 'LEFT'],
              num_states=4*12,
              num_actions=4,
              epochs=1500,
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

        Gt = 0
        step = 0
        while not done:
            next_state, reward, done = env.step(action)
            reward_sum += reward
            next_action = egreedy_policy(q, next_state, exploration_rate)
            Gt += gamma ** step * reward

            state = next_state
            action = next_action
            step += 1

            if i % 100 == 0:
                env.render(q, action=actions[action], colorize_q=True)
            if step % 20 == 0:
                break

        error = Gt - q[state][action]
        q[state][action] += learning_rate * error

        reward_sum_list.append(reward_sum)
        if i % 3 == 0:
            print('Average scores = ', np.mean(reward_sum_list))
            reward_sum_list = []

    return q

def train():
    env = GridWorld()
    q = sarsa_mc(env, render=False, learning_rate=0.5, gamma=0.99)

if __name__ == '__main__':
    train()



