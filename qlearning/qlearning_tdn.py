import numpy as np
from cliffwalk.cliffwalk import GridWorld

def egreedy_policy(q_values, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_values[state])

def qlearning_tdn(env, n_step=4,
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

        step = 0
        reward_ = 0
        while not done:
            step += 1
            action = egreedy_policy(q, state, exploration_rate)
            next_state, reward, done = env.step(action)
            reward_sum += reward
            if step % n_step != 0 and not done:
                reward_ += gamma ** (step-1) * reward
            else:
                td_target = reward_ + gamma ** (step-1) * reward + gamma ** step * np.max(q[next_state])
                td_error = td_target - q[state][action]
                q[state][action] += learning_rate * td_error
                step = 0
                reward_ = 0

            state = next_state

            if i % 100 == 0:
                env.render(q, action=actions[action], colorize_q=True)

        reward_sum_list.append(reward_sum)
        if i % 3 == 0:
            print('Average scores = ', np.mean(reward_sum_list))
            reward_sum_list = []

    return q

def train():
    env = GridWorld()
    q = qlearning_tdn(env, n_step=3, render=False, learning_rate=0.5, gamma=0.99)

if __name__ == '__main__':
    train()



